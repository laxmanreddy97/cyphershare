package com.example.CypherSight

import android.content.Context
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel

class ModelEncoder(context: Context, modelPath: String) {
    private val interpreter: Interpreter

    // Update this from 200704 down to 50176 (1 * 14 * 14 * 256)
    private val TOTAL_ELEMENTS = 50176

    init {
        val options = Interpreter.Options().apply { numThreads = 2 }
        interpreter = Interpreter(loadModelFile(context, modelPath), options)
    }

    private fun loadModelFile(context: Context, path: String): MappedByteBuffer {
        val fileDescriptor = context.assets.openFd(path)
        val inputStream = FileInputStream(fileDescriptor.fileDescriptor)
        return inputStream.channel.map(
            FileChannel.MapMode.READ_ONLY,
            fileDescriptor.startOffset,
            fileDescriptor.declaredLength
        )
    }

    fun encode(inputBuffer: ByteBuffer): FloatArray {
        // Allocate a ByteBuffer for the output (4 bytes per float)
        val outputBuffer = ByteBuffer.allocateDirect(TOTAL_ELEMENTS * 4)
        outputBuffer.order(ByteOrder.nativeOrder())

        // TFLite will write the 4D tensor flatly into this buffer
        interpreter.run(inputBuffer, outputBuffer)

        // Read the buffer into a standard flat Kotlin FloatArray
        outputBuffer.rewind()
        val latentOutput = FloatArray(TOTAL_ELEMENTS)
        outputBuffer.asFloatBuffer().get(latentOutput)

        return latentOutput
    }
}