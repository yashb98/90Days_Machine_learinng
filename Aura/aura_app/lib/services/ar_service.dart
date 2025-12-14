import 'package:flutter/services.dart';

class ArGeospatialService {
  static const platform = MethodChannel('com.aura.app/geospatial');

  Future<void> startNavigation(double lat, double lng) async {
    try {
      await platform.invokeMethod('startNavigation', {
        'latitude': lat,
        'longitude': lng,
      });
    } on PlatformException catch (e) {
      print("Failed to start AR: '${e.message}'.");
    }
  }
}