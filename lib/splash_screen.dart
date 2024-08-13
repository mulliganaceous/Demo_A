import 'package:flutter/material.dart';
import 'package:flutter_svg_animations_with_rive_tutorial/music_player_page.dart';
import 'package:rive/rive.dart';

class SplashScreen extends StatefulWidget {
  const SplashScreen({Key? key}) : super(key: key);

  @override
  _SplashScreenState createState() => _SplashScreenState();
}

class _SplashScreenState extends State<SplashScreen> {
  late RiveAnimationController _controller;

  int level = 200;
  
  @override
  void initState() {
    super.initState();
    _controller = OneShotAnimation(
      'Both',
      autoplay: true,
      onStop: () => setState(() => level = 0),
      onStart: () => setState(() => level = 200),
    );
    Future.delayed(
      Duration(seconds: 5),
      () => Navigator.pushReplacement(
        context,
        MaterialPageRoute(
          builder: (_) => MusicPlayerPage(),
        ),
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    return   Scaffold(
      appBar: AppBar(
        title: const Text('ABC 123'),
      ),
      backgroundColor: Colors.yellow[500],
      body: Center(
        child: Container(
          width: 400,
          child: RiveAnimation.asset(
            'assets/first_animation.riv',
            controllers: [_controller],
          ),
        ),
      ),
    );
  }
}
