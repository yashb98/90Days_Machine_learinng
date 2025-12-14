// package com.example.aura_app

// import android.Manifest
// import android.content.pm.PackageManager
// import android.util.Log
// import androidx.core.app.ActivityCompat
// import androidx.core.content.ContextCompat
// import io.flutter.embedding.android.FlutterActivity
// import io.flutter.embedding.engine.FlutterEngine
// import io.flutter.plugin.common.MethodChannel

// class MainActivity : FlutterActivity() {

//     private lateinit var geoChannel: MethodChannel
//     private var geospatialManager: GeospatialManager? = null
//     private val REQUEST_LOCATION_PERMISSION = 1001
//     private val TAG = "AuraMainActivity"

//     override fun configureFlutterEngine(flutterEngine: FlutterEngine) {
//         super.configureFlutterEngine(flutterEngine)

//         try {
//             // Initialize GeospatialManager with error handling
//             geospatialManager =
//                     try {
//                         GeospatialManager(this)
//                     } catch (e: Exception) {
//                         Log.e(TAG, "Failed to initialize GeospatialManager: ${e.message}")
//                         null
//                     }

//             // Setup MethodChannel
//             geoChannel =
//                     MethodChannel(flutterEngine.dartExecutor.binaryMessenger, "aura/geospatial")

//             geoChannel.setMethodCallHandler { call, result ->
//                 try {
//                     when (call.method) {
//                         "startLocationTracking" -> {
//                             if (geospatialManager == null) {
//                                 result.error(
//                                         "INIT_FAILED",
//                                         "GeospatialManager not initialized",
//                                         null
//                                 )
//                                 return@setMethodCallHandler
//                             }
//                             checkAndRequestLocationPermissions()
//                             result.success("Location tracking initiated")
//                         }
//                         "stopLocationTracking" -> {
//                             geospatialManager?.stopLocationUpdates()
//                             result.success("Location tracking stopped")
//                         }
//                         "getLastPose" -> {
//                             val pose = geospatialManager?.getLastLocation()
//                             result.success(pose ?: mapOf("error" to "No location available yet"))
//                         }
//                         else -> result.notImplemented()
//                     }
//                 } catch (e: Exception) {
//                     Log.e(TAG, "Error handling method: ${call.method}", e)
//                     result.error("METHOD_ERROR", e.message, null)
//                 }
//             }

//             Log.d(TAG, "MainActivity initialized successfully")
//         } catch (e: Exception) {
//             Log.e(TAG, "Fatal error in configureFlutterEngine", e)
//         }
//     }

//     private fun checkAndRequestLocationPermissions() {
//         try {
//             val fine =
//                     ContextCompat.checkSelfPermission(
//                             this,
//                             Manifest.permission.ACCESS_FINE_LOCATION
//                     )
//             val coarse =
//                     ContextCompat.checkSelfPermission(
//                             this,
//                             Manifest.permission.ACCESS_COARSE_LOCATION
//                     )

//             if (fine != PackageManager.PERMISSION_GRANTED ||
//                             coarse != PackageManager.PERMISSION_GRANTED
//             ) {
//                 ActivityCompat.requestPermissions(
//                         this,
//                         arrayOf(
//                                 Manifest.permission.ACCESS_FINE_LOCATION,
//                                 Manifest.permission.ACCESS_COARSE_LOCATION
//                         ),
//                         REQUEST_LOCATION_PERMISSION
//                 )
//             } else {
//                 startLocationUpdatesToFlutter()
//             }
//         } catch (e: Exception) {
//             Log.e(TAG, "Error checking permissions", e)
//         }
//     }

//     private fun startLocationUpdatesToFlutter() {
//         try {
//             geospatialManager?.startLocationUpdates { location ->
//                 val payload =
//                         mapOf(
//                                 "lat" to location.latitude,
//                                 "lng" to location.longitude,
//                                 "accuracy" to location.accuracy,
//                                 "altitude" to location.altitude,
//                                 "timestamp" to location.time
//                         )
//                 try {
//                     geoChannel.invokeMethod("onLocationUpdate", payload)
//                 } catch (e: Exception) {
//                     Log.e(TAG, "Error invoking location update", e)
//                 }
//             }
//         } catch (e: Exception) {
//             Log.e(TAG, "Error starting location updates", e)
//         }
//     }

//     override fun onRequestPermissionsResult(
//             requestCode: Int,
//             permissions: Array<out String>,
//             grantResults: IntArray
//     ) {
//         super.onRequestPermissionsResult(requestCode, permissions, grantResults)

//         if (requestCode == REQUEST_LOCATION_PERMISSION) {
//             if (grantResults.isNotEmpty() && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
//                 startLocationUpdatesToFlutter()
//             } else {
//                 Log.w(TAG, "Location permission denied")
//             }
//         }
//     }

//     override fun onDestroy() {
//         try {
//             geospatialManager?.close()
//         } catch (e: Exception) {
//             Log.e(TAG, "Error closing GeospatialManager", e)
//         }
//         super.onDestroy()
//     }
// }
