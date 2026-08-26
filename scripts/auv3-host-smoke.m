#import <AudioToolbox/AudioToolbox.h>
#import <AVFoundation/AVFoundation.h>
#import <Foundation/Foundation.h>

static int exercise_component(OSType subtype, NSString *label) {
    AudioComponentDescription description = {
        .componentType = kAudioUnitType_Effect,
        .componentSubType = subtype,
        .componentManufacturer = 'Dnze',
        .componentFlags = 0,
        .componentFlagsMask = 0,
    };
    if (AudioComponentFindNext(NULL, &description) == NULL) {
        fprintf(stderr, "component %s was not registered\n", label.UTF8String);
        return 1;
    }

    dispatch_semaphore_t completed = dispatch_semaphore_create(0);
    __block AVAudioUnit *node = nil;
    __block NSError *instantiation_error = nil;
    [AVAudioUnit instantiateWithComponentDescription:description
                                             options:kAudioComponentInstantiation_LoadOutOfProcess
                                   completionHandler:^(AVAudioUnit *audio_unit, NSError *error) {
        node = audio_unit;
        instantiation_error = error;
        dispatch_semaphore_signal(completed);
    }];
    if (dispatch_semaphore_wait(
            completed, dispatch_time(DISPATCH_TIME_NOW, 30 * NSEC_PER_SEC)) != 0) {
        fprintf(stderr, "component %s instantiation timed out\n", label.UTF8String);
        return 1;
    }
    if (node == nil || instantiation_error != nil) {
        fprintf(stderr, "component %s failed to instantiate: %s\n", label.UTF8String,
                instantiation_error.localizedDescription.UTF8String ?: "unknown error");
        return 1;
    }

    AUAudioUnit *unit = node.AUAudioUnit;
    unit.maximumFramesToRender = 512;
    NSError *allocation_error = nil;
    if (![unit allocateRenderResourcesAndReturnError:&allocation_error]) {
        fprintf(stderr, "component %s failed to allocate: %s\n", label.UTF8String,
                allocation_error.localizedDescription.UTF8String ?: "unknown error");
        return 1;
    }
    NSDictionary *state = unit.fullState;
    if (state == nil) {
        fprintf(stderr, "component %s returned no full state\n", label.UTF8String);
        [unit deallocateRenderResources];
        return 1;
    }
    unit.fullState = state;
    [unit reset];
    [unit deallocateRenderResources];
    if (unit.renderResourcesAllocated) {
        fprintf(stderr, "component %s did not release render resources\n", label.UTF8String);
        return 1;
    }

    printf("DENOIZE_AUV3_SMOKE component=%s instantiated=true allocated=true state_round_trip=true teardown=true\n",
           label.UTF8String);
    return 0;
}

int main(void) {
    @autoreleasepool {
        int status = exercise_component('Dn01', @"Dn01");
        status |= exercise_component('Dn02', @"Dn02");
        if (status == 0) {
            puts("Result: AUv3 AVFoundation host smoke passed");
        }
        return status;
    }
}
