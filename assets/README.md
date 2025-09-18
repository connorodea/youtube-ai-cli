# 🎨 Custom Assets Directory

This directory contains custom assets for enhancing your YouTube videos with professional overlays, transitions, sound effects, and background music.

## 📁 Directory Structure

### 🖼️ `overlays/`
Place custom overlay images and video files here:
- **PNG/JPEG overlays**: Logo watermarks, frames, borders, decorative elements
- **MP4 overlays**: Animated logos, particles, light effects
- **Recommended formats**: PNG (with transparency), MP4 (with alpha channel)
- **Resolution**: 1920x1080 or higher for best quality

**Examples:**
```
assets/overlays/
├── logo_watermark.png
├── vintage_frame.png
├── particles_overlay.mp4
├── light_leak.mp4
└── tech_grid_overlay.png
```

### 🎬 `transitions/`
Place custom transition video files here:
- **MP4 transition clips**: Wipes, slides, morphs, creative transitions
- **Duration**: 0.5-2.0 seconds recommended
- **Resolution**: 1920x1080 to match your video
- **Alpha channel**: Supported for complex transitions

**Examples:**
```
assets/transitions/
├── smooth_slide.mp4
├── circle_wipe.mp4
├── glitch_transition.mp4
├── zoom_blur.mp4
└── ink_splatter.mp4
```

### 🔊 `sound_effects/`
Place transition and overlay sound effects here:
- **WAV/MP3 audio files**: Whooshes, impacts, swooshes, glitches
- **Duration**: 0.1-3.0 seconds for transitions
- **Quality**: 44.1kHz, 16-bit minimum
- **Purpose**: Sync with video transitions and overlays

**Examples:**
```
assets/sound_effects/
├── whoosh_01.wav
├── impact_soft.wav
├── glitch_zap.wav
├── slide_whoosh.mp3
└── tech_beep.wav
```

### 🎵 `background_music/`
Place background music tracks here:
- **MP3/WAV files**: Instrumental music, ambient sounds, loops
- **Duration**: 30 seconds to 10+ minutes
- **Quality**: 44.1kHz, 16-bit minimum
- **Volume**: Will be automatically adjusted (typically -20dB)
- **Copyright**: Use only royalty-free or owned music

**Examples:**
```
assets/background_music/
├── upbeat_corporate.mp3
├── ambient_tech.wav
├── inspiring_piano.mp3
├── chill_lo_fi.mp3
└── energetic_electronic.wav
```

### 🔤 `fonts/`
Place custom font files here:
- **TTF/OTF fonts**: Custom typography for text overlays
- **Usage**: Enhanced titles, captions, watermarks
- **License**: Ensure commercial use rights

**Examples:**
```
assets/fonts/
├── Roboto-Bold.ttf
├── Montserrat-Regular.ttf
├── OpenSans-SemiBold.ttf
└── custom_brand_font.otf
```

## 🚀 Usage in CLI Commands

### Auto-YouTube with Custom Assets
```bash
# Use custom assets automatically
python src/youtube_ai/cli/main.py create auto-youtube \
  --topic "Python basics" \
  --style educational \
  --custom-overlay "logo_watermark.png" \
  --custom-transition "smooth_slide.mp4" \
  --background-music "upbeat_corporate.mp3" \
  --transition-sound "whoosh_01.wav"
```

### Professional Video with Full Customization
```bash
# Maximum customization
python src/youtube_ai/cli/main.py create auto-youtube \
  --topic "Advanced AI techniques" \
  --style professional \
  --custom-overlay "tech_grid_overlay.png" \
  --overlay-opacity 0.3 \
  --custom-transition "glitch_transition.mp4" \
  --transition-sound "glitch_zap.wav" \
  --background-music "ambient_tech.wav" \
  --music-volume 0.2
```

## 📋 Asset Guidelines

### ✅ Recommended Specifications
- **Video Resolution**: 1920x1080 (Full HD)
- **Video Framerate**: 30fps
- **Audio Quality**: 44.1kHz, 16-bit
- **Image Formats**: PNG (transparency), JPEG, WEBP
- **Video Formats**: MP4 (H.264), MOV, WEBM
- **Audio Formats**: WAV (best), MP3, AAC

### 🎯 Best Practices
1. **Overlays**: Keep them subtle (20-40% opacity)
2. **Transitions**: Keep them short (0.5-1.5 seconds)
3. **Sound Effects**: Match the visual transition timing
4. **Background Music**: Choose music that matches your content style
5. **File Naming**: Use descriptive names (e.g., `tech_whoosh_fast.wav`)

### 🚫 Avoid
- Copyrighted music without permission
- Overly loud sound effects
- Distracting overlays that cover important content
- Transitions longer than 3 seconds
- Poor quality/pixelated assets

## 🎨 Asset Categories by Style

### Educational
- Clean, minimal overlays
- Smooth, professional transitions
- Soft whooshes and clicks
- Calm, inspiring background music

### Professional
- Corporate logos and frames
- Polished slide transitions  
- Professional sound effects
- Business/corporate music

### Tech/Modern
- Glitch and digital overlays
- Tech-style transitions
- Electronic sound effects
- Synthetic/electronic music

### Creative/Artistic
- Artistic frames and textures
- Creative wipes and morphs
- Unique sound designs
- Experimental music

Start adding your custom assets to these directories, and the YouTube AI CLI will automatically detect and use them in your video generation!