package com.example.CypherSight

import android.graphics.Bitmap
import android.graphics.ImageDecoder
import android.net.Uri
import android.os.Build
import android.os.Bundle
import android.provider.MediaStore
import android.util.Base64
import androidx.activity.ComponentActivity
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.activity.result.launch
import androidx.compose.foundation.Image
import androidx.compose.foundation.layout.*
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.CameraAlt
import androidx.compose.material.icons.filled.PhotoLibrary
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.async
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.RequestBody.Companion.toRequestBody
import org.json.JSONObject
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.security.SecureRandom
import java.util.UUID
import java.util.concurrent.TimeUnit
import javax.crypto.Cipher
import javax.crypto.spec.IvParameterSpec
import javax.crypto.spec.SecretKeySpec

import com.example.CypherSight.ui.theme.CypherSightTheme

class MainActivity : ComponentActivity() {

    private val modelEncoder by lazy { ModelEncoder(this, "encoder_head.tflite") }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContent {
            CypherSightTheme {
                Surface(
                    modifier = Modifier.fillMaxSize(),
                    color = MaterialTheme.colorScheme.background
                ) {
                    RealInferenceScreen(modelEncoder)
                }
            }
        }
    }
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun RealInferenceScreen(modelEncoder: ModelEncoder) {
    val context = LocalContext.current
    val coroutineScope = rememberCoroutineScope()

    // Replace with your actual server IPs
    val thirdPartyUrl = "http://192.168.213.67:8000/process_features"
    val trustedOwnerUrl = "http://192.168.213.67:8001/register_share"

    // Configured with extended timeouts to prevent queuing drops
    val httpClient = remember {
        OkHttpClient.Builder()
            .connectTimeout(30, TimeUnit.SECONDS)
            .writeTimeout(30, TimeUnit.SECONDS)
            .readTimeout(60, TimeUnit.SECONDS)
            .build()
    }

    var resultText by remember { mutableStateOf("Select an image to begin.") }
    var selectedBitmap by remember { mutableStateOf<Bitmap?>(null) }
    var isProcessing by remember { mutableStateOf(false) }

    val imagePickerLauncher = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.GetContent()
    ) { uri: Uri? ->
        if (uri != null) {
            try {
                val bitmap = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.P) {
                    val source = ImageDecoder.createSource(context.contentResolver, uri)
                    ImageDecoder.decodeBitmap(source) { decoder, _, _ ->
                        decoder.allocator = ImageDecoder.ALLOCATOR_SOFTWARE
                    }
                } else {
                    @Suppress("DEPRECATION")
                    MediaStore.Images.Media.getBitmap(context.contentResolver, uri)
                }

                selectedBitmap = Bitmap.createScaledBitmap(bitmap, 224, 224, true)
                resultText = "Ready for secure inference."
            } catch (e: Exception) {
                resultText = "Failed to load image: ${e.message}"
            }
        }
    }

    val cameraLauncher = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.TakePicturePreview()
    ) { bitmap: Bitmap? ->
        if (bitmap != null) {
            selectedBitmap = Bitmap.createScaledBitmap(bitmap, 224, 224, true)
            resultText = "Ready for secure inference."
        } else {
            resultText = "Camera capture cancelled."
        }
    }

    Scaffold(
        topBar = {
            TopAppBar(
                title = { Text("Secure Inference", fontWeight = FontWeight.Bold) },
                colors = TopAppBarDefaults.topAppBarColors(
                    containerColor = MaterialTheme.colorScheme.primaryContainer,
                    titleContentColor = MaterialTheme.colorScheme.onPrimaryContainer
                )
            )
        }
    ) { paddingValues ->
        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(paddingValues)
                .padding(24.dp),
            horizontalAlignment = Alignment.CenterHorizontally
        ) {
            // Framed Image Area
            Card(
                modifier = Modifier
                    .size(260.dp)
                    .padding(8.dp),
                elevation = CardDefaults.cardElevation(defaultElevation = 4.dp),
                colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surfaceVariant)
            ) {
                Box(contentAlignment = Alignment.Center, modifier = Modifier.fillMaxSize()) {
                    if (selectedBitmap != null) {
                        Image(
                            bitmap = selectedBitmap!!.asImageBitmap(),
                            contentDescription = "Selected Image",
                            modifier = Modifier.fillMaxSize(),
                            contentScale = ContentScale.Crop
                        )
                    } else {
                        Icon(
                            imageVector = Icons.Default.PhotoLibrary,
                            contentDescription = "Placeholder",
                            modifier = Modifier.size(64.dp),
                            tint = MaterialTheme.colorScheme.onSurfaceVariant.copy(alpha = 0.5f)
                        )
                    }
                }
            }

            Spacer(modifier = Modifier.height(24.dp))

            // Action Buttons for Image Selection
            Row(
                horizontalArrangement = Arrangement.spacedBy(16.dp),
                modifier = Modifier.fillMaxWidth(),
                verticalAlignment = Alignment.CenterVertically
            ) {
                OutlinedButton(
                    onClick = { cameraLauncher.launch() },
                    modifier = Modifier.weight(1f),
                    enabled = !isProcessing
                ) {
                    Icon(Icons.Default.CameraAlt, contentDescription = "Camera", modifier = Modifier.size(18.dp))
                    Spacer(Modifier.width(8.dp))
                    Text("Camera")
                }

                OutlinedButton(
                    onClick = { imagePickerLauncher.launch("image/*") },
                    modifier = Modifier.weight(1f),
                    enabled = !isProcessing
                ) {
                    Icon(Icons.Default.PhotoLibrary, contentDescription = "Gallery", modifier = Modifier.size(18.dp))
                    Spacer(Modifier.width(8.dp))
                    Text("Gallery")
                }
            }

            Spacer(modifier = Modifier.height(32.dp))

            // Status Text
            Text(
                text = resultText,
                style = MaterialTheme.typography.bodyLarge,
                textAlign = TextAlign.Center,
                color = if (resultText.startsWith("Network") || resultText.startsWith("Failed"))
                    MaterialTheme.colorScheme.error
                else MaterialTheme.colorScheme.onSurface
            )

            Spacer(modifier = Modifier.weight(1f))

            // Main Processing Button
            Button(
                onClick = {
                    val bitmapToProcess = selectedBitmap ?: return@Button
                    isProcessing = true
                    resultText = "Running local processing..."

                    coroutineScope.launch(Dispatchers.IO) {
                        try {
                            // 1. Run local client head
                            val inputBuffer = bitmapToByteBuffer(bitmapToProcess)
                            val latentVector = modelEncoder.encode(inputBuffer)

                            withContext(Dispatchers.Main) {
                                resultText = "Applying Cryptographic Noise..."
                            }

                            // 2. Generate Deterministic Cryptographic Noise
                            val sharedKey = "CryptoAI_SecretKey_256bit_32Byte".toByteArray(Charsets.UTF_8)
                            val ivBytes = ByteArray(16)
                            SecureRandom().nextBytes(ivBytes)
                            val ivBase64 = Base64.encodeToString(ivBytes, Base64.NO_WRAP)

                            val requestId = UUID.randomUUID().toString()

                            val cipher = Cipher.getInstance("AES/CTR/NoPadding")
                            cipher.init(Cipher.ENCRYPT_MODE, SecretKeySpec(sharedKey, "AES"), IvParameterSpec(ivBytes))

                            val zeroBytes = ByteArray(latentVector.size * 4)
                            val keystream = cipher.doFinal(zeroBytes)
                            val keystreamBuffer = ByteBuffer.wrap(keystream).order(ByteOrder.LITTLE_ENDIAN)

                            val jsonArray = org.json.JSONArray()
                            for (i in latentVector.indices) {
                                val randomInt = keystreamBuffer.int
                                val noiseFloat = (randomInt.toFloat() / 2147483647f) * 2.0f
                                val maskedValue = latentVector[i] + noiseFloat
                                jsonArray.put(maskedValue.toDouble())
                            }

                            withContext(Dispatchers.Main) {
                                resultText = "Distributing Secret Shares..."
                            }

                            val tpResponseData = kotlinx.coroutines.coroutineScope {
                                // 3. FIRE SHARE B TO TRUSTED OWNER
                                val ownerDeferred = async(Dispatchers.IO) {
                                    val ownerJson = JSONObject().apply {
                                        put("request_id", requestId)
                                        put("iv", ivBase64)
                                    }
                                    val request = Request.Builder()
                                        .url(trustedOwnerUrl)
                                        .header("Connection", "close") // Fix for router changes
                                        .post(ownerJson.toString().toRequestBody("application/json; charset=utf-8".toMediaType()))
                                        .build()

                                    httpClient.newCall(request).execute().use {
                                        if (!it.isSuccessful) throw Exception("Owner rejected Share B (HTTP ${it.code})")
                                    }
                                }

                                // 4. FIRE SHARE A TO THIRD-PARTY
                                val tpDeferred = async(Dispatchers.IO) {
                                    val tpJson = JSONObject().apply {
                                        put("request_id", requestId)
                                        put("features", jsonArray)
                                    }
                                    val request = Request.Builder()
                                        .url(thirdPartyUrl)
                                        .header("Connection", "close") // Fix for router changes
                                        .post(tpJson.toString().toRequestBody("application/json; charset=utf-8".toMediaType()))
                                        .build()

                                    httpClient.newCall(request).execute().use { response ->
                                        if (!response.isSuccessful) throw Exception("TP Server returned code ${response.code}")
                                        response.body?.string() ?: throw Exception("Empty response body")
                                    }
                                }

                                val tpData = tpDeferred.await()
                                ownerDeferred.await()
                                tpData
                            }

                            val responseJson = JSONObject(tpResponseData)
                            val className = responseJson.optString("prediction", "Unknown")
                            val confidence = responseJson.optDouble("confidence", 0.0)
                            val percentage = String.format("%.1f%%", confidence * 100)

                            withContext(Dispatchers.Main) {
                                resultText = "Prediction: $className\nConfidence: $percentage"
                            }

                        } catch (e: Exception) {
                            withContext(Dispatchers.Main) {
                                resultText = "Network/Processing Error:\n${e.message}"
                            }
                        } finally {
                            withContext(Dispatchers.Main) {
                                isProcessing = false
                            }
                        }
                    }
                },
                enabled = selectedBitmap != null && !isProcessing,
                modifier = Modifier
                    .fillMaxWidth()
                    .height(56.dp)
            ) {
                if (isProcessing) {
                    CircularProgressIndicator(
                        color = MaterialTheme.colorScheme.onPrimary,
                        modifier = Modifier.size(24.dp),
                        strokeWidth = 2.dp
                    )
                    Spacer(Modifier.width(12.dp))
                    Text("Processing securely...")
                } else {
                    Text("Process via Servers", style = MaterialTheme.typography.titleMedium)
                }
            }
        }
    }
}

fun bitmapToByteBuffer(bitmap: Bitmap): ByteBuffer {
    val bufferSize = 224 * 224 * 3 * 4
    val buffer = ByteBuffer.allocateDirect(bufferSize)
    buffer.order(ByteOrder.nativeOrder())

    val intValues = IntArray(224 * 224)
    bitmap.getPixels(intValues, 0, bitmap.width, 0, 0, bitmap.width, bitmap.height)

    for (pixelValue in intValues) {
        val r = ((pixelValue shr 16 and 0xFF) - 127.5f) / 127.5f
        val g = ((pixelValue shr 8 and 0xFF) - 127.5f) / 127.5f
        val b = ((pixelValue and 0xFF) - 127.5f) / 127.5f

        buffer.putFloat(r)
        buffer.putFloat(g)
        buffer.putFloat(b)
    }

    buffer.rewind()
    return buffer
}