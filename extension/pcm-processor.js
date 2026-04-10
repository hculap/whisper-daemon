/**
 * AudioWorklet processor — buffers audio samples and posts Float32Array
 * chunks to the main thread for WebSocket transmission.
 *
 * Runs on the audio rendering thread. process() receives 128 samples
 * per call at the AudioContext's sample rate (16kHz). We buffer to
 * 4096 samples (~256ms) before posting to reduce message overhead.
 */

class PCMProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this._bufferSize = 4096;
    this._buffer = new Float32Array(this._bufferSize);
    this._bytesWritten = 0;
  }

  process(inputs, outputs) {
    const input = inputs[0];
    if (!input || !input[0]) return true;

    const channelData = input[0];

    // Pass through to output for playback (Chrome mutes tab on capture)
    const output = outputs[0];
    if (output && output[0]) {
      output[0].set(channelData);
    }

    for (let i = 0; i < channelData.length; i++) {
      this._buffer[this._bytesWritten++] = channelData[i];

      if (this._bytesWritten >= this._bufferSize) {
        this.port.postMessage(this._buffer);
        this._buffer = new Float32Array(this._bufferSize);
        this._bytesWritten = 0;
      }
    }

    return true;
  }
}

registerProcessor("pcm-processor", PCMProcessor);
