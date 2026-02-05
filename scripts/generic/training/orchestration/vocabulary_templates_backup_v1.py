#!/usr/bin/env python3
"""
BACKUP - Original Vocabulary Templates (v1)
=============================================

This file contains the ORIGINAL templates from vocabulary_generator.py
before the expansion to 100+ action templates, 50-60 pose templates,
and 50-60 expression templates.

Backup Date: 2025-12-06
Original File: vocabulary_generator.py

Template Counts:
- POSE_TEMPLATES: 27 templates
- EXPRESSION_TEMPLATES: 35 templates
- ACTION_TEMPLATES: 35 templates
- STYLE_VARIATIONS: 14 variations
"""

# ============================================================================
# ORIGINAL POSE TEMPLATES (27 templates)
# ============================================================================

POSE_TEMPLATES_V1 = [
    # Standing poses - detailed (5)
    "{character}, {style}, full body view standing straight with neutral expression, clean studio lighting",
    "{character}, {style}, confident stance with arms crossed over chest, three-quarter view, dramatic rim lighting",
    "{character}, {style}, casual pose standing with hands in pockets, relaxed posture, soft ambient lighting",
    "{character}, {style}, standing with one hand on hip in relaxed pose, side profile view, warm natural lighting",
    "{character}, {style}, formal standing pose with arms at sides, front view portrait, even studio lighting",

    # Sitting poses - detailed (5)
    "{character}, {style}, sitting on chair with legs crossed, upper body view, soft diffused lighting",
    "{character}, {style}, sitting on ground with legs extended forward, full body shot, outdoor natural lighting",
    "{character}, {style}, sitting cross-legged in meditative pose, peaceful expression, gentle backlight, serene atmosphere",
    "{character}, {style}, sitting with knees pulled up in relaxed casual pose, intimate framing, warm lighting",
    "{character}, {style}, sitting at desk in focused working pose, medium shot, task lighting from above",

    # Action poses - detailed (6)
    "{character}, {style}, walking forward with natural gait and movement, dynamic full body shot, cinematic lighting",
    "{character}, {style}, running in dynamic athletic motion, action pose mid-stride, dramatic lighting with motion blur effect",
    "{character}, {style}, jumping through air in mid-flight pose, energetic expression, dynamic camera angle, vibrant lighting",
    "{character}, {style}, reaching upward with arms stretched high, inspiring pose, low angle shot, heroic lighting",
    "{character}, {style}, leaning forward examining something closely, curious expression, focused lighting, intimate character study",
    "{character}, {style}, leaning back in relaxed comfortable pose, casual demeanor, soft ambient light",

    # Camera angles - detailed (6)
    "{character}, {style}, full body portrait shot from front view, standing pose, neutral expression, professional studio lighting, clean background",
    "{character}, {style}, full body three-quarter view showing depth and dimension, dynamic pose, cinematic lighting setup, high quality render",
    "{character}, {style}, complete side profile view full body, clear silhouette, rim lighting, artistic composition, detailed modeling",
    "{character}, {style}, full body back view showing rear perspective, over-shoulder glance, dramatic lighting, professional 3d work",
    "{character}, {style}, full body low angle shot looking up at character, heroic framing, powerful lighting, epic composition",
    "{character}, {style}, full body high angle shot looking down, unique perspective, soft overhead lighting, interesting framing",

    # Upper body - detailed (4)
    "{character}, {style}, upper body portrait with arms relaxed at sides, friendly expression, soft key lighting, professional headshot quality",
    "{character}, {style}, upper body shot with arms raised expressively, animated gesture, dynamic lighting, vibrant character personality",
    "{character}, {style}, close-up upper body with hands clasped on chest, emotional pose, gentle lighting, intimate character moment",
    "{character}, {style}, upper body view looking back over shoulder, engaging eye contact, beautiful rim lighting, compelling composition",

    # Dynamic poses - detailed (5)
    "{character}, {style}, turning around in dynamic rotating motion, action pose, motion lighting effects, energetic 3d animation",
    "{character}, {style}, bending down reaching for object on ground, realistic motion, natural lighting, physically accurate pose",
    "{character}, {style}, kneeling on one knee in heroic pose, determined expression, dramatic uplighting, cinematic framing",
    "{character}, {style}, lying down in relaxed resting pose, peaceful scene, soft diffused lighting, calm atmosphere",
    "{character}, {style}, dancing with expressive flowing movement, joyful energy, colorful stage lighting, dynamic composition",
]

# ============================================================================
# ORIGINAL EXPRESSION TEMPLATES (35 templates)
# ============================================================================

EXPRESSION_TEMPLATES_V1 = [
    # Basic emotions - happy (4)
    "{character}, {style}, close-up portrait showing genuine happy expression with bright smile and joyful eyes, warm studio lighting, detailed facial features",
    "{character}, {style}, very happy character laughing heartily with mouth open wide and eyes crinkled in joy, expressive animation, natural lighting, pixar style emotional performance",
    "{character}, {style}, upper body shot with beaming smile showing teeth, cheerful expression radiating happiness, soft key lighting",
    "{character}, {style}, headshot portrait with gentle smile and warm friendly expression, subtle eye sparkle, even lighting, detailed 3d character model",

    # Sad expressions (3)
    "{character}, {style}, close-up showing sad melancholic expression with downcast eyes and slight frown, moody dramatic lighting, emotional depth, cinematic quality render",
    "{character}, {style}, portrait of character with sorrowful expression and glistening eyes suggesting tears, gentle rim lighting, touching emotional moment",
    "{character}, {style}, upper body view with dejected posture and sad face looking down, subdued lighting, expressive character performance",

    # Angry expressions (3)
    "{character}, {style}, intense close-up with angry expression featuring furrowed brows and stern glare, dramatic contrast lighting, powerful emotion, detailed facial animation",
    "{character}, {style}, headshot showing frustrated angry face with clenched jaw and narrowed eyes, hard lighting, strong character emotion",
    "{character}, {style}, medium shot with aggressive angry stance and fierce facial expression, dynamic lighting, intense character moment, cinematic animation style",

    # Surprised expressions (3)
    "{character}, {style}, close-up portrait with shocked surprised expression featuring wide eyes and open mouth, bright frontal lighting, dramatic reaction shot, pixar animation quality",
    "{character}, {style}, headshot showing astonished face with raised eyebrows and gasping mouth, clear studio lighting, expressive character animation, detailed 3d model",
    "{character}, {style}, upper body view capturing moment of surprise with startled wide-eyed expression, dynamic lighting, authentic emotional response, high detail render",

    # Scared/worried expressions (3)
    "{character}, {style}, close-up showing fearful scared expression with worried eyes and tense face, shadowy dramatic lighting, emotional intensity, cinematic 3d animation",
    "{character}, {style}, portrait with anxious nervous expression featuring wide fearful eyes and uncertain mouth, moody lighting, character vulnerability, detailed animation",
    "{character}, {style}, headshot of frightened character with trembling worried look and pale complexion, eerie lighting, emotional performance",

    # Neutral/calm expressions (3)
    "{character}, {style}, professional neutral headshot with calm serene expression and relaxed facial features, even studio lighting, clean character render",
    "{character}, {style}, close-up portrait showing peaceful tranquil expression with soft eyes and gentle mouth, natural lighting, meditative character moment, high detail",
    "{character}, {style}, upper body shot with composed neutral face and balanced features, soft ambient lighting, professional character animation, smooth 3d render",

    # Thoughtful/contemplative (3)
    "{character}, {style}, close-up showing thoughtful contemplative expression with hand on chin and pensive eyes, warm side lighting, intellectual character moment, detailed 3d",
    "{character}, {style}, portrait of character deep in thought with furrowed brow and distant gaze, soft key lighting, introspective expression, cinematic animation quality",
    "{character}, {style}, headshot with reflective contemplative face and thoughtful eyes looking aside, gentle lighting, character depth, high quality render",

    # Curious expressions (3)
    "{character}, {style}, close-up with curious inquisitive expression featuring raised eyebrows and interested eyes, bright lighting, engaging character animation, detailed 3d model",
    "{character}, {style}, portrait showing eager curious face with wide attentive eyes and slight smile, natural lighting, charming expression",
    "{character}, {style}, upper body view with questioning curious look and tilted head, soft lighting, expressive character performance",

    # Confused expressions (2)
    "{character}, {style}, headshot showing confused puzzled expression with scrunched brows and uncertain mouth, even lighting, relatable character moment",
    "{character}, {style}, close-up of bewildered confused face with questioning eyes and perplexed look, natural lighting, authentic emotion, detailed 3d render",

    # Embarrassed/shy (2)
    "{character}, {style}, portrait with embarrassed blushing expression featuring averted eyes and shy smile, warm lighting, endearing character moment",
    "{character}, {style}, close-up showing shy timid expression with downward glance and subtle blush, soft lighting, gentle emotion, high quality animation",

    # Confident/determined (2)
    "{character}, {style}, powerful headshot with confident determined expression featuring strong gaze and firm jaw, dramatic lighting, heroic character moment, detailed 3d",
    "{character}, {style}, close-up showing self-assured confident face with direct eye contact and slight smirk, bold lighting, character strength, photorealistic render",

    # Excited expressions (2)
    "{character}, {style}, vibrant portrait with excited enthusiastic expression featuring bright eyes and wide smile, energetic lighting, joyful character animation, high detail 3d",
    "{character}, {style}, close-up of thrilled excited face with animated features and sparkling eyes, dynamic lighting, infectious enthusiasm, pixar quality render",

    # Mischievous/playful (2)
    "{character}, {style}, headshot with mischievous playful grin and knowing wink, warm lighting, charming character personality, detailed animation quality",
    "{character}, {style}, close-up showing impish playful expression with sly smile and twinkling eyes, creative lighting, fun character moment",

    # Loving/warm (2)
    "{character}, {style}, tender portrait with loving affectionate expression featuring soft smile and warm eyes, gentle lighting, heartfelt emotion, cinematic 3d animation",
    "{character}, {style}, close-up showing caring compassionate face with kind eyes and gentle expression, soft key lighting, touching character moment",

    # Tired/sleepy (2)
    "{character}, {style}, headshot with tired exhausted expression featuring heavy eyelids and weary look, subdued lighting, relatable fatigue",
    "{character}, {style}, close-up showing sleepy drowsy face with half-closed eyes and relaxed features, soft lighting, natural tiredness",
]

# ============================================================================
# ORIGINAL ACTION TEMPLATES (35 templates)
# ============================================================================

ACTION_TEMPLATES_V1 = [
    # Sports - basketball (3)
    "{character}, {style}, dynamic action shot playing basketball while dribbling ball with athletic stance, court environment, energetic motion capture, cinematic sports lighting, high quality 3d animation",
    "{character}, {style}, full body view shooting basketball with perfect form and focused expression, gym setting, frozen action moment, dramatic rim lighting, detailed character animation",
    "{character}, {style}, intense basketball play jumping for layup with extended arms and determined face, dynamic camera angle, motion blur effect, professional sports render",

    # Sports - soccer/football (2)
    "{character}, {style}, action pose kicking soccer ball with powerful leg swing and balanced body position, grass field background, dynamic movement, outdoor natural lighting",
    "{character}, {style}, athletic stance dribbling soccer ball with agile footwork and concentrated expression, stadium environment, motion capture quality, cinematic sports photography style",

    # Swimming (2)
    "{character}, {style}, swimming freestyle stroke with arms extended forward and streamlined body position, underwater perspective, aquatic environment, beautiful subsurface lighting, fluid animation",
    "{character}, {style}, dynamic swimming action cutting through water with powerful strokes and athletic form, pool setting, splash effects, dramatic underwater lighting, high detail 3d",

    # Cycling (2)
    "{character}, {style}, riding bicycle while pedaling with proper cycling form and focused expression, outdoor path setting, motion captured movement, natural daylight, detailed character render",
    "{character}, {style}, action shot on bicycle with dynamic riding pose and wind-swept appearance, scenic background, energetic composition, golden hour lighting, cinematic 3d animation",

    # Skateboarding (2)
    "{character}, {style}, performing skateboard trick mid-air with impressive athletic form and concentrated face, skate park environment, frozen action moment, dramatic angle, high quality render",
    "{character}, {style}, skateboarding action with dynamic balance and flowing movement, urban setting, motion blur on wheels, vibrant lighting, professional animation quality",

    # Reading (2)
    "{character}, {style}, quietly reading book while holding it open with both hands and absorbed expression, cozy indoor environment, warm reading light, peaceful atmosphere, detailed 3d render",
    "{character}, {style}, sitting comfortably with book in hands showing engaged reading pose and thoughtful face, library setting, soft ambient lighting, intimate character moment, photorealistic quality",

    # Writing/drawing (3)
    "{character}, {style}, focused on writing while holding pen with proper grip and concentrated expression, desk workspace, task lighting from above, studious atmosphere",
    "{character}, {style}, creative drawing activity with pencil in hand and artistic expression on face, art studio environment, natural lighting, inspiring moment",
    "{character}, {style}, painting on canvas with brush in hand and passionate artistic expression, colorful studio setting, warm lighting, creative energy, detailed character render",

    # Eating/drinking (3)
    "{character}, {style}, eating meal while holding food with natural gesture and enjoyment expression, dining table setting, appetizing atmosphere, warm lighting, realistic 3d animation",
    "{character}, {style}, drinking from cup with both hands wrapped around it and satisfied expression, cafe environment, cozy ambiance, soft lighting, photorealistic character detail",
    "{character}, {style}, cooking at stove preparing food with focused chef-like concentration and engaged posture, kitchen setting, warm cooking light, domestic scene, high quality render",

    # Social gestures - greeting (2)
    "{character}, {style}, waving hello with raised hand and friendly welcoming smile, open body language, bright outdoor setting, cheerful atmosphere, natural lighting, expressive 3d animation",
    "{character}, {style}, giving enthusiastic wave with both arms and joyful greeting expression, energetic pose, warm environment, inviting lighting, charming character moment",

    # Social gestures - gesturing (3)
    "{character}, {style}, pointing at something with extended finger and curious interested expression, engaged body language, clear environment, focused lighting, communicative pose render",
    "{character}, {style}, giving thumbs up gesture with confident smile and positive body language, encouraging atmosphere, bright lighting, affirmative character moment, detailed 3d",
    "{character}, {style}, clapping hands together with applauding motion and appreciative expression, celebratory atmosphere, warm lighting, joyful character animation, high quality",

    # Social interaction - physical (2)
    "{character}, {style}, shaking hands in formal greeting with professional demeanor and friendly smile, business environment, respectful interaction, even lighting",
    "{character}, {style}, giving warm hug with caring embrace and affectionate expression, emotional moment, soft lighting, heartfelt character interaction, cinematic animation quality",

    # Exercise/fitness (4)
    "{character}, {style}, stretching arms overhead in warm-up pose with relaxed athletic form, fitness environment, energizing atmosphere, natural lighting, healthy activity render",
    "{character}, {style}, doing push-ups in proper exercise form with focused determined expression, gym setting, workout lighting, athletic dedication, detailed 3d character",
    "{character}, {style}, performing yoga balance pose with centered calm expression and graceful form, peaceful studio, soft ambient light, meditative atmosphere, high quality animation",
    "{character}, {style}, lifting weights with proper form and concentrated strength training expression, gym environment, motivating lighting, athletic character moment, photorealistic render",

    # Climbing/exploration (3)
    "{character}, {style}, climbing wall with athletic grip and determined upward gaze, adventure setting, dynamic perspective, dramatic lighting, action-packed 3d animation",
    "{character}, {style}, looking around with searching expression and alert observant posture, exploratory environment, curious atmosphere, natural lighting",
    "{character}, {style}, picking up object from ground with bending-down motion and interested expression, narrative action, focused lighting, story-driven pose",

    # Musical performance (3)
    "{character}, {style}, playing guitar while strumming with musical expression and engaged passionate face, performance setting, stage lighting, artistic moment",
    "{character}, {style}, singing with microphone held close and emotional performing expression, concert atmosphere, dramatic stage lights, powerful character performance",
    "{character}, {style}, dancing with flowing rhythmic movement and joyful expressive face, dance floor setting, dynamic colorful lighting, energetic choreography",

    # Environmental interaction (3)
    "{character}, {style}, opening door with reaching hand and anticipatory expression, interior transition moment, natural lighting, narrative action, detailed character animation",
    "{character}, {style}, climbing stairs with upward walking motion and focused forward gaze, architectural setting, ascending perspective, even lighting",
    "{character}, {style}, peeking around corner with cautious curious expression and careful body language, suspenseful moment, dramatic side lighting, engaging character action",
]

# ============================================================================
# ORIGINAL STYLE VARIATIONS (14 variations)
# ============================================================================

STYLE_VARIATIONS_V1 = [
    # Basic Pixar style variations (7)
    "3d animation, pixar style, high quality, detailed",
    "3d animated character, smooth shading, studio lighting",
    "pixar-style 3d render, professional quality",
    "3d cg animation, clean render",
    "animated 3d character, cinematic lighting",
    "high-quality 3d animation, detailed modeling",
    "3d computer graphics, smooth surfaces, realistic lighting",

    # Advanced material-aware variations (7)
    "pixar style 3d animation, smooth shading, PBR materials, subsurface scattering on skin, soft ambient occlusion, professional CGI render",
    "3d animated character, PBR skin materials, detailed subsurface scattering, ambient occlusion, high quality professional render",
    "pixar-style 3d render, physically based rendering, subsurface scattering, soft ambient occlusion, cinematic lighting, detailed character modeling",
    "high-quality 3d animation, PBR materials, realistic skin subsurface scattering, soft ambient occlusion, professional CGI quality",
    "3d cg animation, smooth PBR shading, subsurface scattering effects, ambient occlusion rendering, cinematic quality",
    "animated 3d character, physically based materials, subsurface skin scattering, soft AO, professional animation render",
    "pixar animation style, PBR materials and textures, realistic subsurface scattering, ambient occlusion, high-end CGI production quality",
]


# ============================================================================
# SUMMARY
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("BACKUP - Original Vocabulary Templates (v1)")
    print("=" * 60)
    print(f"\nTemplate Counts:")
    print(f"  POSE_TEMPLATES:       {len(POSE_TEMPLATES_V1)} templates")
    print(f"  EXPRESSION_TEMPLATES: {len(EXPRESSION_TEMPLATES_V1)} templates")
    print(f"  ACTION_TEMPLATES:     {len(ACTION_TEMPLATES_V1)} templates")
    print(f"  STYLE_VARIATIONS:     {len(STYLE_VARIATIONS_V1)} variations")
    print(f"\nTotal: {len(POSE_TEMPLATES_V1) + len(EXPRESSION_TEMPLATES_V1) + len(ACTION_TEMPLATES_V1)} templates")
    print("\nThis backup was created on 2025-12-06")
    print("Original file: vocabulary_generator.py")
