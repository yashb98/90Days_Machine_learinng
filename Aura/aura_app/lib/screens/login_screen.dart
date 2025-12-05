import 'dart:ui'; // Required for Glassmorphism
import 'package:flutter/material.dart';
import 'package:firebase_auth/firebase_auth.dart';

class LoginScreen extends StatefulWidget {
  const LoginScreen({super.key});

  @override
  State<LoginScreen> createState() => _LoginScreenState();
}

class _LoginScreenState extends State<LoginScreen> with SingleTickerProviderStateMixin {
  final _emailController = TextEditingController();
  final _passwordController = TextEditingController();
  bool _isLoading = false;

  // Animation Controller for the "Breathing" Logo
  late AnimationController _logoController;
  late Animation<double> _logoAnimation;

  @override
  void initState() {
    super.initState();
    _logoController = AnimationController(
      vsync: this,
      duration: const Duration(seconds: 3),
    )..repeat(reverse: true);

    _logoAnimation = Tween<double>(begin: 1.0, end: 1.1).animate(
      CurvedAnimation(parent: _logoController, curve: Curves.easeInOut),
    );
  }

  @override
  void dispose() {
    _logoController.dispose();
    _emailController.dispose();
    _passwordController.dispose();
    super.dispose();
  }

  Future<void> _login() async {
    setState(() => _isLoading = true);
    try {
      await FirebaseAuth.instance.signInWithEmailAndPassword(
        email: _emailController.text.trim(),
        password: _passwordController.text.trim(),
      );
    } catch (e) {
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(
            content: Text("Login Error: $e"),
            backgroundColor: Colors.redAccent,
          ),
        );
      }
    } finally {
      if (mounted) setState(() => _isLoading = false);
    }
  }

  Future<void> _loginAnon() async {
    setState(() => _isLoading = true);
    try {
      await FirebaseAuth.instance.signInAnonymously();
    } catch (e) {
      print(e);
    } finally {
      if (mounted) setState(() => _isLoading = false);
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Colors.black,
      body: Stack(
        children: [
          // 1. AMBIENT BACKGROUND
          Positioned(
            top: -100, left: -100,
            child: Container(
              width: 300, height: 300,
              decoration: BoxDecoration(
                color: Colors.blueAccent.withOpacity(0.4),
                shape: BoxShape.circle,
                boxShadow: [BoxShadow(blurRadius: 120, color: Colors.blueAccent.withOpacity(0.5))],
              ),
            ),
          ),
          Positioned(
            bottom: -100, right: -100,
            child: Container(
              width: 300, height: 300,
              decoration: BoxDecoration(
                color: Colors.purpleAccent.withOpacity(0.2),
                shape: BoxShape.circle,
                boxShadow: [BoxShadow(blurRadius: 120, color: Colors.purpleAccent.withOpacity(0.4))],
              ),
            ),
          ),

          // 2. GLASS CARD CONTENT
          Center(
            child: SingleChildScrollView(
              padding: const EdgeInsets.all(24),
              child: ClipRRect(
                borderRadius: BorderRadius.circular(30),
                child: BackdropFilter(
                  filter: ImageFilter.blur(sigmaX: 10, sigmaY: 10),
                  child: Container(
                    padding: const EdgeInsets.all(32),
                    decoration: BoxDecoration(
                      color: Colors.white.withOpacity(0.05),
                      borderRadius: BorderRadius.circular(30),
                      border: Border.all(color: Colors.white.withOpacity(0.1)),
                    ),
                    child: Column(
                      mainAxisAlignment: MainAxisAlignment.center,
                      children: [
                        // Animated Logo
                        ScaleTransition(
                          scale: _logoAnimation,
                          child: Container(
                            padding: const EdgeInsets.all(20),
                            decoration: BoxDecoration(
                              shape: BoxShape.circle,
                              color: Colors.black.withOpacity(0.3),
                              border: Border.all(color: Colors.blueAccent.withOpacity(0.5), width: 2),
                              boxShadow: [
                                BoxShadow(
                                  color: Colors.blueAccent.withOpacity(0.3),
                                  blurRadius: 30,
                                  spreadRadius: 5,
                                )
                              ],
                            ),
                            child: const Icon(Icons.remove_red_eye_rounded, size: 60, color: Colors.blueAccent),
                          ),
                        ),
                        
                        const SizedBox(height: 24),
                        
                        const Text(
                          "AURA",
                          style: TextStyle(
                            color: Colors.white, 
                            fontSize: 32, 
                            fontWeight: FontWeight.w900, 
                            letterSpacing: 4
                          ),
                        ),
                        const SizedBox(height: 8),
                        Text(
                          "VISION INTELLIGENCE",
                          style: TextStyle(
                            color: Colors.blueAccent.withOpacity(0.8), 
                            fontSize: 12, 
                            fontWeight: FontWeight.bold, 
                            letterSpacing: 2
                          ),
                        ),
                        
                        const SizedBox(height: 48),

                        // Modern Input Fields
                        _buildTextField(_emailController, "Email Address", Icons.email_outlined),
                        const SizedBox(height: 16),
                        _buildTextField(_passwordController, "Password", Icons.lock_outline, isObscure: true),

                        const SizedBox(height: 32),

                        // Primary Action Button
                        SizedBox(
                          width: double.infinity,
                          height: 55,
                          child: ElevatedButton(
                            onPressed: _isLoading ? null : _login,
                            style: ElevatedButton.styleFrom(
                              backgroundColor: Colors.blueAccent,
                              foregroundColor: Colors.white,
                              elevation: 10,
                              shadowColor: Colors.blueAccent.withOpacity(0.5),
                              shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(16)),
                            ),
                            child: _isLoading 
                                ? const SizedBox(width: 24, height: 24, child: CircularProgressIndicator(color: Colors.white, strokeWidth: 2)) 
                                : const Text("ACCESS SYSTEM", style: TextStyle(fontSize: 16, fontWeight: FontWeight.bold, letterSpacing: 1)),
                          ),
                        ),
                        
                        const SizedBox(height: 24),
                        
                        // Secondary Action
                        TextButton(
                          onPressed: _loginAnon,
                          style: TextButton.styleFrom(foregroundColor: Colors.grey),
                          child: Row(
                            mainAxisSize: MainAxisSize.min,
                            children: const [
                              Icon(Icons.terminal, size: 16),
                              SizedBox(width: 8),
                              Text("Initialize Guest Protocol", style: TextStyle(fontSize: 14)),
                            ],
                          ),
                        )
                      ],
                    ),
                  ),
                ),
              ),
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildTextField(TextEditingController controller, String label, IconData icon, {bool isObscure = false}) {
    return TextField(
      controller: controller,
      obscureText: isObscure,
      style: const TextStyle(color: Colors.white),
      cursorColor: Colors.blueAccent,
      decoration: InputDecoration(
        prefixIcon: Icon(icon, color: Colors.white70, size: 20),
        labelText: label,
        labelStyle: TextStyle(color: Colors.white.withOpacity(0.5)),
        filled: true,
        fillColor: Colors.black.withOpacity(0.3),
        contentPadding: const EdgeInsets.symmetric(vertical: 20, horizontal: 20),
        enabledBorder: OutlineInputBorder(
          borderRadius: BorderRadius.circular(16),
          borderSide: BorderSide(color: Colors.white.withOpacity(0.1)),
        ),
        focusedBorder: OutlineInputBorder(
          borderRadius: BorderRadius.circular(16),
          borderSide: const BorderSide(color: Colors.blueAccent, width: 1.5),
        ),
      ),
    );
  }
}