import 'dart:ui';
import 'package:flutter/material.dart';
import 'package:geolocator/geolocator.dart';

class StatusBar extends StatelessWidget {
  final Position? currentPosition;
  final bool isConnected;
  final VoidCallback onMenuTap;
  final String currentMode;

  const StatusBar({
    super.key,
    required this.currentPosition,
    required this.isConnected,
    required this.onMenuTap,
    required this.currentMode,
  });

  @override
  Widget build(BuildContext context) {
    return Positioned(
      top: 50, left: 20, right: 20,
      child: ClipRRect(
        borderRadius: BorderRadius.circular(20),
        child: BackdropFilter(
          filter: ImageFilter.blur(sigmaX: 10, sigmaY: 10),
          child: Container(
            padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 12),
            color: Colors.black.withOpacity(0.4),
            child: Row(
              mainAxisAlignment: MainAxisAlignment.spaceBetween,
              children: [
                // 1. HAMBURGER MENU (Left)
                GestureDetector(
                  onTap: onMenuTap,
                  child: const Icon(Icons.menu, color: Colors.white, size: 28),
                ),

                // 2. CURRENT MODE (Center)
                Container(
                  padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 6),
                  decoration: BoxDecoration(
                    color: Colors.white.withOpacity(0.1),
                    borderRadius: BorderRadius.circular(12),
                  ),
                  child: Row(
                    children: [
                      Icon(_getModeIcon(currentMode), color: Colors.white70, size: 14),
                      const SizedBox(width: 6),
                      Text(
                        currentMode.toUpperCase(),
                        style: const TextStyle(
                          color: Colors.white, 
                          fontSize: 12, 
                          fontWeight: FontWeight.bold,
                          letterSpacing: 1
                        ),
                      ),
                    ],
                  ),
                ),

                // 3. ONLINE STATUS (Right)
                Row(
                  children: [
                    // GPS Dot
                    Container(
                      width: 8, height: 8,
                      decoration: BoxDecoration(
                        color: currentPosition != null ? Colors.blueAccent : Colors.grey,
                        shape: BoxShape.circle,
                      ),
                    ),
                    const SizedBox(width: 8),
                    // Online Dot
                    Container(
                      width: 8, height: 8,
                      decoration: BoxDecoration(
                        color: isConnected ? Colors.greenAccent : Colors.redAccent,
                        shape: BoxShape.circle,
                        boxShadow: [
                          BoxShadow(
                            color: (isConnected ? Colors.green : Colors.red).withOpacity(0.6),
                            blurRadius: 8,
                            spreadRadius: 2
                          )
                        ]
                      ),
                    ),
                  ],
                ),
              ],
            ),
          ),
        ),
      ),
    );
  }

  IconData _getModeIcon(String mode) {
    switch (mode) {
      case 'reading': return Icons.menu_book_rounded;
      case 'scenery': return Icons.landscape_rounded;
      default: return Icons.shield_outlined;
    }
  }
}