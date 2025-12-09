import 'package:flutter/material.dart';
import 'package:firebase_auth/firebase_auth.dart';

class MainDrawer extends StatelessWidget {
  final String currentMode;
  final Function(String) onModeSelected;

  const MainDrawer({
    super.key,
    required this.currentMode,
    required this.onModeSelected,
  });

  @override
  Widget build(BuildContext context) {
    final user = FirebaseAuth.instance.currentUser;
    
    return Drawer(
      backgroundColor: const Color(0xFF1E1E1E),
      child: Column(
        children: [
          // Header with User Info
          UserAccountsDrawerHeader(
            decoration: const BoxDecoration(
              gradient: LinearGradient(
                colors: [Colors.blueAccent, Colors.purpleAccent],
                begin: Alignment.topLeft,
                end: Alignment.bottomRight,
              ),
            ),
            accountName: const Text(
              "Aura Vision", 
              style: TextStyle(fontWeight: FontWeight.bold, fontSize: 20)
            ),
            accountEmail: Text(user?.email ?? "Guest User"),
            currentAccountPicture: const CircleAvatar(
              backgroundColor: Colors.white,
              child: Icon(Icons.remove_red_eye, color: Colors.black, size: 30),
            ),
          ),
          
          const Padding(
            padding: EdgeInsets.only(left: 16, top: 10, bottom: 10),
            child: Align(
              alignment: Alignment.centerLeft,
              child: Text(
                "VISION MODES", 
                style: TextStyle(color: Colors.grey, fontWeight: FontWeight.bold)
              ),
            ),
          ),

          // Menu Items
          _buildDrawerItem(context, Icons.shield_outlined, "Safety Guide", "safety", Colors.greenAccent),
          _buildDrawerItem(context, Icons.menu_book_rounded, "Text Reader", "reading", Colors.blueAccent),
          _buildDrawerItem(context, Icons.landscape_rounded, "Scenery Describer", "scenery", Colors.purpleAccent),

          const Divider(color: Colors.grey),
          
          // Logout Option
          ListTile(
            leading: const Icon(Icons.logout, color: Colors.redAccent),
            title: const Text("Logout", style: TextStyle(color: Colors.white)),
            onTap: () async {
              await FirebaseAuth.instance.signOut();
            },
          ),
        ],
      ),
    );
  }

  Widget _buildDrawerItem(BuildContext context, IconData icon, String title, String mode, Color color) {
    bool isSelected = currentMode == mode;
    return ListTile(
      leading: Icon(icon, color: isSelected ? color : Colors.white54),
      title: Text(
        title, 
        style: TextStyle(
          color: isSelected ? Colors.white : Colors.white70, 
          fontWeight: isSelected ? FontWeight.bold : FontWeight.normal
        )
      ),
      trailing: isSelected ? Icon(Icons.check_circle, color: color, size: 18) : null,
      tileColor: isSelected ? color.withOpacity(0.1) : null,
      onTap: () {
        Navigator.pop(context); // Close the drawer
        onModeSelected(mode);   // Trigger mode switch
      },
    );
  }
}