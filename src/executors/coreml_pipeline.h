// Internal feature-provider forwarding only. No tensor data access or arithmetic.
// Included once by the Objective-C++ exception firewall.
@interface RustNNPipelineDescription : NSObject
@property(nonatomic, copy) NSDictionary *inputDescriptionsByName;
@property(nonatomic, copy) NSDictionary *outputDescriptionsByName;
@end
@implementation RustNNPipelineDescription
@end

@interface RustNNPipeline : NSObject
@property(nonatomic, strong) RustNNPipelineDescription *modelDescription;
@property(nonatomic, copy) NSArray<MLModel *> *children;
@property(nonatomic, copy) NSDictionary *plan;
- (id<MLFeatureProvider>)predictionFromFeatures:(id<MLFeatureProvider>)features
    options:(MLPredictionOptions *)options error:(NSError **)error;
- (id<MLFeatureProvider>)predictionFromFeatures:(id<MLFeatureProvider>)features error:(NSError **)error;
@end

static NSError *rustnn_pipeline_error(NSString *message) {
    return [NSError errorWithDomain:@"RustNN.NativePipeline" code:1
        userInfo:@{NSLocalizedDescriptionKey: message}];
}

@implementation RustNNPipeline
- (id<MLFeatureProvider>)predictionFromFeatures:(id<MLFeatureProvider>)features error:(NSError **)error {
    return [self predictionFromFeatures:features options:[[MLPredictionOptions alloc] init] error:error];
}
- (id<MLFeatureProvider>)predictionFromFeatures:(id<MLFeatureProvider>)features
    options:(MLPredictionOptions *)options error:(NSError **)error {
    NSMutableArray *values = [NSMutableArray arrayWithCapacity:[self.plan[@"slots"] unsignedIntegerValue]];
    for (NSUInteger i=0; i<[self.plan[@"slots"] unsignedIntegerValue]; ++i) [values addObject:NSNull.null];
    for (NSDictionary *binding in self.plan[@"inputs"]) {
        MLFeatureValue *value = [features featureValueForName:binding[@"name"]];
        if (!value) { if (error) *error=rustnn_pipeline_error(@"Missing pipeline input"); return nil; }
        values[[binding[@"slot"] unsignedIntegerValue]] = value;
    }
    NSUInteger index=0;
    // Keep errors strong across each child's autorelease pool. Writing an
    // autoreleasing out-parameter inside that pool would leave the caller with
    // a dangling NSError before ARC performs its writeback.
    NSError *failure=nil;
    for (NSDictionary *child in self.plan[@"children"]) {
        @autoreleasepool {
            NSMutableDictionary *selected = [NSMutableDictionary dictionary];
            for (NSDictionary *binding in child[@"inputs"]) {
                id value = values[[binding[@"slot"] unsignedIntegerValue]];
                if (value == NSNull.null) { failure=rustnn_pipeline_error(@"Released pipeline input"); break; }
                selected[binding[@"name"]] = value;
            }
            if (failure) break;
            MLDictionaryFeatureProvider *input = [[MLDictionaryFeatureProvider alloc] initWithDictionary:selected error:&failure];
            if (!input) { if (!failure) failure=rustnn_pipeline_error(@"Failed to construct child input"); break; }
            // Only final destinations belong to the caller. Passing every
            // backing to every child lets duplicate names alias earlier values.
            NSMutableDictionary *backings = [NSMutableDictionary dictionary];
            for (NSDictionary *binding in child[@"outputs"]) {
                for (NSDictionary *final in self.plan[@"outputs"]) {
                    if ([final[@"slot"] isEqual:binding[@"slot"]] && options.outputBackings[final[@"name"]]) {
                        backings[binding[@"name"]] = options.outputBackings[final[@"name"]];
                    }
                }
            }
            MLPredictionOptions *childOptions = [[MLPredictionOptions alloc] init];
            childOptions.outputBackings = backings;
            id<MLFeatureProvider> output = [self.children[index++] predictionFromFeatures:input options:childOptions error:&failure];
            if (!output) { if (!failure) failure=rustnn_pipeline_error(@"Child returned no prediction"); break; }
            for (NSDictionary *binding in child[@"outputs"]) {
                MLFeatureValue *value = [output featureValueForName:binding[@"name"]];
                if (!value) { failure=rustnn_pipeline_error(@"Missing child output"); break; }
                values[[binding[@"slot"] unsignedIntegerValue]] = value;
            }
            for (NSNumber *slot in child[@"release"]) values[slot.unsignedIntegerValue] = NSNull.null;
        }
        if (failure) break;
    }
    if (failure) { if (error) *error=failure; return nil; }
    NSMutableDictionary *final = [NSMutableDictionary dictionary];
    for (NSDictionary *binding in self.plan[@"outputs"]) {
        id value=values[[binding[@"slot"] unsignedIntegerValue]];
        if (value == NSNull.null) { if (error) *error=rustnn_pipeline_error(@"Missing pipeline output"); return nil; }
        final[binding[@"name"]]=value;
    }
    return [[MLDictionaryFeatureProvider alloc] initWithDictionary:final error:error];
}
@end

static RustNNPipeline *rustnn_load_pipeline(NSURL *url, MLModelConfiguration *config,
                                           NSDictionary *plan, NSError **error,
                                           bool (*progress)(void *, size_t, size_t, bool), void *context) {
    RustNNPipeline *pipeline = [[RustNNPipeline alloc] init];
    NSMutableArray<MLModel *> *models = [NSMutableArray array];
    NSMutableDictionary *inputDescriptions = [NSMutableDictionary dictionary];
    NSMutableDictionary *slotDescriptions = [NSMutableDictionary dictionary];
    NSError *failure=nil;
    size_t childIndex=0, childCount=[plan[@"children"] count];
    for (NSDictionary *child in plan[@"children"]) {
        if (progress && !progress(context, childIndex, childCount, false)) {
            failure=rustnn_pipeline_error(@"Child-load progress callback failed"); break;
        }
        @autoreleasepool {
            MLModel *model = [MLModel modelWithContentsOfURL:[url URLByAppendingPathComponent:child[@"path"]]
                configuration:config error:&failure];
            if (!model) { if (!failure) failure=rustnn_pipeline_error(@"Child returned no model"); break; }
            [models addObject:model];
            for (NSDictionary *binding in child[@"inputs"]) {
                MLFeatureDescription *desc=model.modelDescription.inputDescriptionsByName[binding[@"name"]];
                if (!desc) { failure=rustnn_pipeline_error(@"Child input description differs"); break; }
                if (!slotDescriptions[binding[@"slot"]]) slotDescriptions[binding[@"slot"]]=desc;
            }
            // Report the first mismatch; the output check must not replace it.
            if (failure) break;
            for (NSDictionary *binding in child[@"outputs"]) {
                MLFeatureDescription *desc=model.modelDescription.outputDescriptionsByName[binding[@"name"]];
                if (!desc) { failure=rustnn_pipeline_error(@"Child output description differs"); break; }
                slotDescriptions[binding[@"slot"]]=desc;
            }
        }
        if (failure) break;
        if (progress && !progress(context, childIndex, childCount, true)) {
            failure=rustnn_pipeline_error(@"Child-load progress callback failed"); break;
        }
        ++childIndex;
    }
    if (failure) { if (error) *error=failure; return nil; }
    // Descriptions come from the children. A pipeline input that no child
    // consumes has none, yet may be returned unchanged as a pipeline output;
    // callers then bind that feature with its WebNN operand type.
    NSMutableSet *inputSlots = [NSMutableSet set];
    for (NSDictionary *binding in plan[@"inputs"]) {
        [inputSlots addObject:binding[@"slot"]];
        id desc=slotDescriptions[binding[@"slot"]];
        if (desc) inputDescriptions[binding[@"name"]]=desc;
    }
    NSMutableDictionary *outputDescriptions = [NSMutableDictionary dictionary];
    for (NSDictionary *binding in plan[@"outputs"]) {
        id desc=slotDescriptions[binding[@"slot"]];
        if (desc) outputDescriptions[binding[@"name"]]=desc;
        else if (![inputSlots containsObject:binding[@"slot"]]) {
            if (error) *error=rustnn_pipeline_error(@"No native description for pipeline output"); return nil;
        }
    }
    pipeline.modelDescription=[[RustNNPipelineDescription alloc] init];
    pipeline.modelDescription.inputDescriptionsByName=inputDescriptions;
    pipeline.modelDescription.outputDescriptionsByName=outputDescriptions;
    pipeline.children=models;
    pipeline.plan=plan;
    return pipeline;
}
