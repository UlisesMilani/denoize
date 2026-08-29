package io.github.penguin425.denoize.sdk

import androidx.test.ext.junit.runners.AndroidJUnit4
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Assert.assertThrows
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith

@RunWith(AndroidJUnit4::class)
class DenoizeSdkInstrumentedTest {
    @Test
    fun nativeProcessorPreservesLengthAndTypedCancellation() {
        val processor = DenoizeProcessor.create(
            DenoizeOptions(
                sampleRate = 48_000,
                channels = 1,
                frameSize = 256,
                maxFramesPerCall = 256,
                maxBufferedFrames = 1_024,
            ),
        )
        val input = FloatArray(256) { index -> ((index % 17) - 8) / 32.0f }
        val processed = processor.processInterleaved(input)
        val tail = processor.finish()
        assertEquals(input.size, processed.size + tail.size)
        assertTrue((processed.asSequence() + tail.asSequence()).all(Float::isFinite))

        processor.reset()
        processor.cancellation.cancel()
        val error = assertThrows(DenoizeSdkException::class.java) {
            processor.processInterleaved(input)
        }
        assertEquals(DenoizeStatus.CANCELLED, error.status)
        processor.reset()
        processor.close()
    }

    @Test
    fun routeChangesAndSuspensionNeverResumeStaleState() {
        val first = AudioRoute(sampleRate = 48_000, bufferFrames = 256, channels = 1)
        val second = AudioRoute(sampleRate = 44_100, bufferFrames = 512, channels = 2)
        val session = DenoizeMobileSession(frameSize = 256)
        session.configure(first)
        assertEquals(1, session.routeGeneration)
        session.start()
        session.onRouteChanged(second)
        assertEquals(MobileSessionState.READY, session.state)
        assertEquals(2, session.routeGeneration)
        assertEquals(second, session.route)
        session.start()
        session.onBackgrounded()
        assertEquals(MobileSessionState.BACKGROUNDED, session.state)
        assertNull(session.route)
        session.resume(second)
        assertEquals(3, session.routeGeneration)
        assertEquals(MobileSessionState.READY, session.state)
        session.close()
    }
}
