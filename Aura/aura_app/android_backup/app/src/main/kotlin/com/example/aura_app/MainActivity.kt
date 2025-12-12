// // package com.example.aura_app

// // import io.flutter.embedding.android.FlutterActivity 
// // import io.flutter.embedding.engine.FlutterEngine
// // import io.flutter.plugin.common.MethodChannel

// // class MainActivity : FlutterActivity() {

// //     private lateinit var geoChannel: MethodChannel

// //     override fun configureFlutterEngine(flutterEngine: FlutterEngine) {
// //         super.configureFlutterEngine(flutterEngine)

// //         geoChannel = MethodChannel(flutterEngine.dartExecutor.binaryMessenger, "aura/geospatial")
// //     }

// //     // You will call this later from your ARCore code when you have pose values
// //     fun sendPoseToFlutter(
// //             lat: Double,
// //             lng: Double,
// //             alt: Double,
// //             heading: Double,
// //             hAcc: Double,
// //             headingAcc: Double
// //     ) {
// //         val data =
// //                 mapOf(
// //                         "lat" to lat,
// //                         "lng" to lng,
// //                         "alt" to alt,
// //                         "heading" to heading,
// //                         "hAcc" to hAcc,
// //                         "headingAcc" to headingAcc,
// //                 )
// //         geoChannel.invokeMethod("geospatialPose", data)
// //     }
// // }

// package com.example.aura_app

// import io.flutter.embedding.android.FlutterActivity   // <-- add this import
// import io.flutter.embedding.engine.FlutterEngine
// import io.flutter.plugin.common.MethodChannel

// class MainActivity : FlutterActivity() {

//     private lateinit var geoChannel: MethodChannel

//     override fun configureFlutterEngine(flutterEngine: FlutterEngine) {
//         super.configureFlutterEngine(flutterEngine)

//         geoChannel = MethodChannel(
//             flutterEngine.dartExecutor.binaryMessenger,
//             "aura/geospatial"
//         )
//     }

//     // You will call this later from your ARCore code when you have pose values
//     fun sendPoseToFlutter(
//         lat: Double,
//         lng: Double,
//         alt: Double,
//         heading: Double,
//         hAcc: Double,
//         headingAcc: Double
//     ) {
//         val data = mapOf(
//             "lat" to lat,
//             "lng" to lng,
//             "alt" to alt,
//             "heading" to heading,
//             "hAcc" to hAcc,
//             "headingAcc" to headingAcc,
//         )
//         geoChannel.invokeMethod("geospatialPose", data)
//     }
// }
package com.example.aura_app

import io.flutter.embedding.android.FlutterActivity

class MainActivity : FlutterActivity()

