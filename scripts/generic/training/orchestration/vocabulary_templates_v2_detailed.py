#!/usr/bin/env python3
"""
Vocabulary Templates v2.0 - DETAILED VERSION
=============================================

Each template is designed to produce 100-120 token prompts when combined with
character prefix and style tags.

Template structure:
- Core pose/expression/action description (detailed)
- Body mechanics and positioning
- Camera angle and framing
- Environment/setting
- Lighting description
- Technical rendering quality

Target: 60-80 tokens per template (raw)
Full prompt with character + style: 100-120 tokens
"""

# ============================================================================
# POSE TEMPLATES (66 templates) - DETAILED VERSION
# ============================================================================

POSE_TEMPLATES = [
    # === STANDING POSES (15) ===
    "{character}, {style}, full body view standing straight in natural relaxed posture with weight evenly distributed on both feet, arms hanging loosely at sides with slight natural curve, neutral calm expression with soft gaze directed at camera, professional studio environment with seamless backdrop, three-point lighting setup with soft key light creating gentle shadows and subtle rim light separating figure from background, smooth skin rendering with subsurface scattering, detailed fabric textures on clothing",

    "{character}, {style}, confident power stance with arms crossed firmly over chest and feet shoulder-width apart, weight shifted slightly back with chin lifted showing self-assured attitude, three-quarter view from front-left angle showing depth and dimension, dramatic rim lighting from behind creating strong silhouette edge, key light from front highlighting facial features, professional studio setting with dark gradient background, high quality character render with detailed clothing folds",

    "{character}, {style}, casual relaxed pose standing with both hands tucked into pockets, shoulders slightly rounded in comfortable posture, weight shifted to one leg creating natural hip tilt, friendly approachable expression with slight smile, soft diffused ambient lighting from multiple directions creating minimal shadows, warm color temperature suggesting comfortable indoor environment, full body framing with generous negative space, detailed fabric physics on casual clothing",

    "{character}, {style}, standing pose with one hand resting confidently on hip and opposite arm relaxed at side, weight shifted creating elegant contrapposto stance, side profile view showing clear silhouette and body proportions, warm natural lighting streaming from side suggesting golden hour conditions, outdoor environment with soft bokeh background, rim light highlighting hair and shoulder edges, professional character portrait with detailed skin texture and natural pose",

    "{character}, {style}, formal upright standing pose with arms positioned neatly at sides and hands relaxed, perfect vertical posture with shoulders back and chin level, direct front view showing symmetrical body positioning, even professional studio lighting with balanced key and fill lights eliminating harsh shadows, clean minimalist background, corporate professional atmosphere, high quality render with crisp details and smooth gradients",

    "{character}, {style}, dynamic ready stance with one foot slightly forward and knees softly bent, arms held in alert position away from body, engaged expression showing anticipation and focus, low camera angle looking up at figure creating powerful heroic framing, dramatic lighting with strong key light from above and intense rim light, action-ready atmosphere with energetic mood, detailed athletic wear with realistic material properties",

    "{character}, {style}, relaxed lean pose with back against invisible wall support, arms crossed casually over chest with one ankle crossed over the other, cool confident expression with slight knowing smile, three-quarter angle showing dimensional depth, urban environment lighting with mixed warm and cool sources, lifestyle photography aesthetic with natural candid feeling, detailed casual streetwear clothing with realistic fabric draping",

    "{character}, {style}, elegant contrapposto stance inspired by classical sculpture with weight on one leg creating flowing S-curve through body, one arm relaxed and other slightly raised in graceful gesture, refined serene expression, artistic side lighting creating dramatic chiaroscuro effect with deep shadows and bright highlights, fine art photography aesthetic, sophisticated composition with careful attention to negative space and body lines",

    "{character}, {style}, assertive power pose with hands firmly planted on hips and elbows angled outward, chest forward and chin raised in dominant confident posture, direct challenging eye contact with camera, bold dramatic lighting with strong contrast between lit and shadow areas, powerful atmosphere suggesting leadership and authority, professional corporate environment, detailed business attire with crisp fabric rendering",

    "{character}, {style}, respectful formal pose with hands clasped together behind back, feet together in attentive stance, dignified composed expression with direct but soft gaze, even balanced lighting creating professional portrait aesthetic, institutional or formal environment setting, conservative color palette, clean precise rendering with attention to posture details and clothing presentation",

    "{character}, {style}, thoughtful contemplative standing pose with weight shifted to back foot, one hand raised to chin in thinking gesture, head slightly tilted with introspective distant expression, soft moody lighting with gentle shadows suggesting quiet reflective moment, minimalist environment with muted background, intimate personal atmosphere, detailed rendering of subtle facial expression and hand positioning",

    "{character}, {style}, energetic celebratory pose with arms raised high above head in victory gesture, weight on balls of feet suggesting upward momentum, joyful triumphant expression with wide genuine smile, dynamic low angle view emphasizing upward energy, bright vivid lighting with warm tones suggesting positive achievement moment, festive atmosphere with sense of accomplishment, dynamic clothing movement suggesting recent action",

    "{character}, {style}, protective guarded stance with arms wrapped around self in self-comforting embrace, shoulders slightly hunched and body angled away, vulnerable uncertain expression with downcast eyes, soft diffused lighting with cool blue undertones creating somber mood, quiet isolated environment, emotionally sensitive moment captured with care, subtle body language details showing psychological state",

    "{character}, {style}, curious investigative pose leaning forward with one hand shading eyes as if looking into distance, weight on front foot with back leg extended, alert interested expression with focused gaze, outdoor natural lighting with bright directional sunlight creating clear shadows, adventure exploration setting, dynamic composition with sense of movement and discovery, detailed outdoor clothing with practical gear elements",

    "{character}, {style}, welcoming open pose with arms spread wide in friendly greeting gesture, weight balanced with slight forward lean showing approachability, warm genuine smile with crinkled eyes, bright cheerful lighting with soft wraparound quality, friendly social environment, inviting atmosphere that draws viewer in, casual comfortable clothing with relaxed fabric draping",

    # === SITTING POSES (12) ===
    "{character}, {style}, sitting cross-legged on floor in comfortable meditation pose with hands resting gently on knees, straight spine with relaxed shoulders, peaceful serene expression with closed or softly focused eyes, soft diffused overhead lighting creating gentle shadows, tranquil minimalist environment suggesting quiet contemplative space, calming atmosphere with sense of inner peace, loose comfortable clothing with natural fabric draping",

    "{character}, {style}, sitting in chair with legs elegantly crossed at knee, one arm resting on armrest and other hand in lap, refined confident expression with subtle smile, professional studio lighting with key light from front-right creating dimensional modeling, sophisticated office or formal environment, polished professional atmosphere, detailed business attire with crisp pressed appearance",

    "{character}, {style}, casual seated pose on ground with legs extended straight forward, weight supported on arms positioned behind body, relaxed friendly expression looking at camera, outdoor natural lighting with warm afternoon sun creating long shadows, grass or outdoor setting suggesting casual leisure time, laid-back weekend atmosphere, comfortable casual clothing with natural wrinkles and folds",

    "{character}, {style}, seated pose with knees drawn up to chest and arms wrapped around legs, compact protective posture with chin resting on knees, thoughtful contemplative or slightly vulnerable expression, soft intimate lighting with warm tones, cozy indoor environment suggesting personal private moment, emotionally nuanced atmosphere, comfortable home clothing with soft fabric textures",

    "{character}, {style}, focused working pose seated at desk leaning forward with intense concentration, hands positioned for typing or writing, serious engaged expression with furrowed brow, practical task lighting from above with desk lamp quality, productive office or study environment, intellectual working atmosphere, professional work attire with rolled sleeves suggesting active engagement",

    "{character}, {style}, relaxed lounging pose leaning back in chair with casual sprawl, one leg extended and other bent, completely at-ease expression with lazy contentment, warm ambient lighting suggesting comfortable evening setting, cozy living space environment, restful leisure atmosphere, comfortable loungewear with relaxed draping",

    "{character}, {style}, attentive seated pose perched on edge of seat leaning forward with engaged interest, hands on knees in ready position, alert curious expression with wide attentive eyes, bright even lighting suggesting active social or educational setting, engaged learning or meeting environment, participatory atmosphere showing active listening, neat casual clothing appropriate for social setting",

    "{character}, {style}, graceful side-saddle seated pose with legs positioned elegantly to one side, refined upright posture with hands folded in lap, poised sophisticated expression, soft glamorous lighting with gentle highlights and shadows, elegant refined environment, classic timeless aesthetic, formal feminine attire with flowing fabric details",

    "{character}, {style}, hunched forward seated pose with elbows resting on knees and head bowed, weight collapsed forward in tired or dejected posture, weary or troubled expression partially hidden, moody dramatic lighting with strong shadows, quiet isolated environment, heavy emotional atmosphere suggesting burden or exhaustion, wrinkled disheveled clothing reflecting mental state",

    "{character}, {style}, seated pose on high stool or bar chair with legs dangling and feet on lower rung, casual confident posture with slight lean, friendly approachable expression, lifestyle photography lighting with natural mixed sources, casual social environment like cafe or bar, relaxed social atmosphere, trendy casual clothing with modern styling",

    "{character}, {style}, formal kneeling pose sitting back on heels with straight spine, hands resting formally on thighs, composed dignified expression, clean even lighting suggesting formal or ceremonial setting, traditional or cultural environment, respectful formal atmosphere, traditional or formal attire appropriate to setting",

    "{character}, {style}, playful seated pose sitting with legs to one side and weight supported on one arm, other hand gesturing expressively, animated cheerful expression mid-conversation, bright natural lighting with warm friendly quality, casual social environment, fun lighthearted atmosphere, colorful casual clothing with youthful styling",

    # === DYNAMIC ACTION POSES (12) ===
    "{character}, {style}, walking pose captured mid-stride with one foot forward and arms swinging naturally in opposition, balanced weight transfer shown in body lean, purposeful focused expression, outdoor environment with directional lighting creating motion shadows, urban or natural pathway setting, sense of forward movement and destination, casual practical clothing with natural movement wrinkles",

    "{character}, {style}, running pose frozen mid-sprint with dynamic leg extension and powerful arm drive, body leaning forward into motion, determined athletic expression with focused intensity, action sports lighting with dramatic key light freezing motion, outdoor track or path environment, energetic athletic atmosphere, performance sportswear with visible motion dynamics",

    "{character}, {style}, jumping pose captured at peak height with body fully extended and arms reaching upward, legs tucked or extended in flight, exhilarated joyful expression, dynamic low angle view emphasizing height and energy, bright outdoor lighting with rim light highlighting airborne figure, open sky or elevated environment, weightless floating moment captured, athletic clothing with fabric catching air",

    "{character}, {style}, reaching pose stretching upward with body fully extended, arms stretched high and weight on toes, aspirational determined expression gazing upward, inspiring low angle composition, dramatic lighting with strong uplighting, motivational achievement atmosphere, figure silhouetted against bright background, fitted athletic wear showing body extension",

    "{character}, {style}, twisting turning pose with torso rotated and hips facing different direction than shoulders, dynamic spiral energy through body, engaged alert expression in direction of turn, action photography lighting capturing motion energy, dynamic environment suggesting activity, pivoting movement frozen in time, flexible athletic clothing accommodating motion range",

    "{character}, {style}, bending down pose reaching toward ground with knees bent and back curved, weight shifted forward over bent knees, concentrated focused expression on task, natural practical lighting, everyday environment suggesting common activity, functional movement pose, practical casual clothing with natural stretch and movement",

    "{character}, {style}, heroic kneeling pose on one knee with opposite leg forward in classic hero landing, one arm down for balance and other arm raised, powerful determined expression, dramatic epic lighting with strong rim light and atmospheric effects, cinematic action environment, powerful triumphant atmosphere, heroic costume or athletic wear with dramatic details",

    "{character}, {style}, peaceful lying pose on back with body relaxed and arms at sides or on chest, completely relaxed muscles and serene expression, soft overhead lighting with gentle diffusion, comfortable resting environment, restful calm atmosphere, comfortable loose clothing with natural draping",

    "{character}, {style}, expressive dancing pose mid-movement with flowing arm gestures and shifted weight, dynamic body lines creating sense of rhythm and motion, joyful liberated expression, colorful stage or club lighting with dramatic effects, performance or celebration environment, musical energetic atmosphere, flowing movement-friendly clothing with dynamic motion",

    "{character}, {style}, crouching pose in low athletic stance with bent knees and lowered center of gravity, weight on balls of feet ready for action, alert watchful expression with focused intensity, dramatic tension lighting with strong shadows, suspenseful environment suggesting anticipation, ready-to-spring athletic energy, practical action-ready clothing",

    "{character}, {style}, stretching pose with one arm reaching across body and opposite direction lean, full body extension in lateral stretch, refreshed rejuvenated expression, morning or exercise lighting with natural quality, wellness environment suggesting self-care, healthy active atmosphere, comfortable exercise wear with stretch properties",

    "{character}, {style}, balancing pose on one leg with opposite leg extended and arms out for stability, focused concentration on maintaining equilibrium, determined concentrated expression, clean studio lighting emphasizing body lines and balance, minimalist environment focusing on figure, precise controlled atmosphere, fitted clothing showing body alignment",

    # === CAMERA ANGLE POSES (12) ===
    "{character}, {style}, dramatic low angle full body shot looking up at standing figure creating powerful imposing presence, figure looming large in frame, confident commanding expression looking down at camera, strong heroic lighting from above and behind, epic cinematic atmosphere, powerful dominant composition, detailed costume or outfit rendered from below perspective",

    "{character}, {style}, high angle full body shot looking down at figure creating sense of vulnerability or approachability, figure appearing smaller in expansive environment, open receptive expression looking up at camera, soft overhead lighting with gentle quality, environmental context visible around figure, intimate accessible atmosphere, relaxed natural clothing seen from above",

    "{character}, {style}, extreme close-up facial portrait filling frame with detailed features, intimate personal framing showing eyes nose and mouth, intense direct emotional expression engaging viewer, beautiful soft portrait lighting with catchlights in eyes, shallow depth of field isolating face, deeply personal atmosphere, every skin texture and facial detail visible",

    "{character}, {style}, full body side profile silhouette view showing clear outline and proportions, clean distinct shape against contrasting background, dignified neutral expression in profile, strong backlight creating rim around entire figure, minimalist artistic composition, timeless classic atmosphere, clean clothing lines contributing to silhouette",

    "{character}, {style}, over-the-shoulder view from behind showing back of head and shoulder with glimpse of environment ahead, voyeuristic following perspective, partially visible expression in three-quarter profile, natural environmental lighting, sense of journey or destination, narrative storytelling atmosphere, detailed rendering of hair and clothing from behind",

    "{character}, {style}, dutch angle tilted camera view creating dynamic tension and energy, figure positioned along diagonal lines, dynamic alert expression matching angular composition, dramatic film noir lighting with strong shadows, suspenseful or action atmosphere, stylized cinematic composition, clothing details emphasized by dramatic angle",

    "{character}, {style}, worms eye view from ground level looking up dramatically at towering figure, extreme low perspective emphasizing height and power, dominant confident expression from above, sky or ceiling visible behind figure, monumental impressive scale, powerful larger-than-life atmosphere, detailed costume visible from unique angle",

    "{character}, {style}, birds eye view looking straight down at figure from directly above, unique top-down perspective showing floor pattern and figure shape, figure looking up at camera creating connection, soft even overhead lighting, geometric composition with figure in environment, artistic abstract quality, interesting patterns in clothing and surroundings",

    "{character}, {style}, medium close-up upper body portrait from chest to head, comfortable conversational framing distance, warm engaged expression with natural eye contact, professional portrait lighting with soft key and fill, neutral or softly blurred background, approachable professional atmosphere, detailed clothing visible in upper body",

    "{character}, {style}, full body environmental portrait showing figure in context of larger setting, figure positioned using rule of thirds in expansive scene, contemplative or observational expression, natural environmental lighting appropriate to setting, strong sense of place and atmosphere, narrative context visible, appropriate clothing for environment",

    "{character}, {style}, reflection shot showing figure and their mirror or water reflection, doubled composition with interesting symmetry, introspective expression gazing at reflection, soft ambient lighting with reflective surface quality, contemplative self-aware atmosphere, metaphorical depth to image, clothing visible in both direct and reflected view",

    "{character}, {style}, foreground framing shot with figure seen through or beside environmental elements, natural frame created by architecture foliage or objects, candid unposed expression as if unaware of camera, documentary natural lighting, voyeuristic discovered moment feeling, authentic genuine atmosphere, natural clothing appropriate to setting",

    # === UPPER BODY POSES (9) ===
    "{character}, {style}, professional headshot upper body portrait with shoulders and head in frame, straight-on business-appropriate pose, confident professional expression with subtle smile, classic portrait lighting with key light at 45 degrees, clean neutral background, corporate professional atmosphere, detailed business attire visible at shoulders and collar",

    "{character}, {style}, upper body shot with expressive hand gestures near face or chest, animated communicative pose capturing mid-conversation moment, engaged enthusiastic expression, lifestyle lighting with natural mixed sources, social conversational environment suggested, warm friendly atmosphere, casual stylish clothing with interesting details",

    "{character}, {style}, upper body portrait with arms crossed in confident power pose, squared shoulders facing camera, assertive self-assured expression, dramatic lighting emphasizing strength, professional corporate environment, leadership authority atmosphere, tailored professional attire with structured shoulders",

    "{character}, {style}, casual upper body shot with one hand running through hair or touching face, relaxed candid pose suggesting genuine moment, natural easy-going expression, soft natural window lighting, intimate personal environment, authentic relaxed atmosphere, comfortable casual clothing with soft textures",

    "{character}, {style}, upper body shot leaning forward toward camera with engaging presence, elbows on surface creating forward lean, interested attentive expression focused on viewer, warm inviting lighting, personal conversation framing, intimate connected atmosphere, neat casual clothing appropriate for close interaction",

    "{character}, {style}, upper body portrait with chin resting thoughtfully on hand, contemplative intellectual pose, pensive thoughtful expression with distant gaze, moody atmospheric lighting with soft shadows, quiet reflective environment, introspective academic atmosphere, professional casual attire suggesting intellectual setting",

    "{character}, {style}, energetic upper body shot with fist pump or celebratory arm gesture, dynamic victorious pose capturing triumph moment, elated joyful expression with wide smile, bright energetic lighting, exciting achievement environment suggested, triumphant celebratory atmosphere, sporty or casual clothing for active celebration",

    "{character}, {style}, artistic upper body portrait with interesting arm placement creating visual flow, posed compositional arrangement of limbs, serene artistic expression, dramatic chiaroscuro lighting with strong contrast, fine art photography aesthetic, timeless elegant atmosphere, simple clothing allowing focus on form",

    "{character}, {style}, emotional upper body portrait with hands clasped together in pleading or hopeful gesture, vulnerable open pose, earnest sincere expression with emotional intensity, soft sympathetic lighting, intimate emotional environment, deeply felt moment captured, simple undistracting clothing focusing attention on emotion",

    # === EXPRESSIVE BODY LANGUAGE (6) ===
    "{character}, {style}, surprised startled pose with body jumped back and hands raised defensively, weight shifted away from surprise source, wide-eyed shocked expression with raised eyebrows and open mouth, dramatic moment lighting with strong contrast, unexpected encounter environment, high tension surprise atmosphere, clothing showing sudden movement disruption",

    "{character}, {style}, frustrated exasperated pose with hands thrown up or on head, tense body language showing irritation, annoyed frustrated expression with furrowed brow, harsh direct lighting matching mood, stressful overwhelming environment suggested, palpable frustration atmosphere, disheveled clothing reflecting emotional state",

    "{character}, {style}, shy bashful pose with body partially turned away and shoulders hunched, self-conscious protective body language, embarrassed expression with averted eyes and slight blush coloring, soft gentle lighting with warm tones, intimate social environment, awkward endearing atmosphere, modest conservative clothing matching personality",

    "{character}, {style}, proud triumphant pose with chest puffed out and head held high, confident accomplished body language, satisfied proud expression with slight smile, grand flattering lighting emphasizing achievement, success recognition environment, accomplished victorious atmosphere, formal or achievement-appropriate attire",

    "{character}, {style}, exhausted drained pose with slumped shoulders and heavy limbs, depleted tired body language, weary exhausted expression with half-closed eyes, dim low-energy lighting, end of long day environment, physically and mentally tired atmosphere, wrinkled worn clothing showing day's activities",

    "{character}, {style}, excited eager pose with bouncing energy and animated gestures, barely contained enthusiasm in body language, delighted excited expression with bright eyes, vibrant energetic lighting, anticipation-filled environment, contagious excitement atmosphere, fun casual clothing with energetic colors",
]


# ============================================================================
# EXPRESSION TEMPLATES (67 templates) - DETAILED VERSION
# ============================================================================

EXPRESSION_TEMPLATES = [
    # === HAPPY EXPRESSIONS (8) ===
    "{character}, {style}, close-up portrait showing genuine happy joyful expression with bright wide smile revealing teeth, eyes crinkled with authentic happiness creating crow's feet, raised cheeks pushing up into eyes, relaxed brow and forehead showing ease, warm studio lighting with soft key light from front-left creating gentle modeling, catchlights sparkling in eyes, smooth skin rendering with subtle subsurface scattering, highly detailed facial features capturing every nuance of joy",

    "{character}, {style}, portrait capturing ecstatic elated expression with head tilted back in hearty laughter, mouth open wide in genuine unrestrained mirth, eyes squeezed nearly shut from intensity of joy, visible laugh lines and dynamic facial movement, bright cheerful lighting with warm color temperature, candid spontaneous moment of pure happiness, natural skin texture with realistic motion",

    "{character}, {style}, gentle content smile portrait with soft closed-lip smile curving naturally upward, peaceful serene eyes with relaxed eyelids, subtle warmth radiating from expression, harmonious balanced features showing inner contentment, soft diffused lighting with minimal shadows creating peaceful mood, calm tranquil atmosphere, smooth detailed skin rendering with healthy natural glow",

    "{character}, {style}, playful mischievous grin portrait with asymmetrical smile showing knowing amusement, one eyebrow slightly raised in playful expression, twinkling eyes suggesting secret joke or plan, dynamic slightly tilted head position, bright fun lighting with warm highlights, lighthearted impish atmosphere, detailed facial features capturing subtle humor",

    "{character}, {style}, beaming proud smile portrait showing wide genuine grin with teeth visible, eyes bright with achievement satisfaction, lifted chin showing confidence, flushed cheeks from emotional warmth, flattering portrait lighting emphasizing positive features, warm proud atmosphere, detailed skin texture showing healthy emotional color",

    "{character}, {style}, shy sweet smile portrait with modest gentle smile and slightly lowered gaze, bashful expression with soft pink blush on cheeks, demure modest demeanor with subtle lip curve, gentle soft lighting creating innocent atmosphere, sweet endearing mood, delicate detailed skin rendering with realistic blush coloring",

    "{character}, {style}, relieved grateful expression portrait showing released tension in relaxed features, eyes showing wetness from emotional relief, thankful appreciative smile, exhausted but happy appearance, soft sympathetic lighting, emotional cathartic atmosphere, detailed facial features showing complex emotion of relief",

    "{character}, {style}, amused entertained expression portrait with suppressed laughter visible in pressed lips, sparkle of humor in eyes, cheeks slightly raised from held-back smile, knowing appreciative expression, bright lively lighting, witty intelligent atmosphere, detailed features capturing subtle amusement",

    # === SAD EXPRESSIONS (6) ===
    "{character}, {style}, deeply sad melancholic portrait with downturned mouth corners and drooping features, glistening eyes wet with unshed tears, heavy eyelids showing emotional weight, furrowed brow expressing inner pain, moody dramatic lighting with deep shadows emphasizing sorrow, somber blue-toned color palette, highly detailed rendering of emotional devastation in facial features",

    "{character}, {style}, quietly sorrowful expression with single tear rolling down cheek, trembling lower lip showing effort to maintain composure, red-rimmed eyes from crying, vulnerable raw emotion visible, soft sympathetic lighting with gentle quality, intimate private grief moment, detailed skin texture showing tear tracks and emotional flush",

    "{character}, {style}, heartbroken devastated expression with face crumpled in emotional anguish, multiple tears streaming down face, mouth open in silent sob or cry, eyes squeezed shut against pain, dramatic emotional lighting, gut-wrenching atmosphere of loss, detailed rendering of intense emotional suffering",

    "{character}, {style}, wistful nostalgic sad expression with distant unfocused gaze, soft sad smile tinged with memory, eyes showing depth of remembered happiness and current loss, gentle resigned features, warm but melancholic lighting, bittersweet atmospheric mood, subtle detailed facial expression capturing complex emotion",

    "{character}, {style}, disappointed let-down expression with fallen features showing crushed expectations, dimmed eyes lacking usual sparkle, slight downward pull to mouth corners, deflated dejected appearance, flat uninspiring lighting matching mood, atmosphere of unfulfilled hopes, detailed features showing subtle disappointment",

    "{character}, {style}, lonely isolated expression with distant hollow gaze, flat affect in facial features, lips pressed in thin line, disconnected withdrawn appearance, cold harsh lighting emphasizing isolation, empty desolate atmosphere, detailed rendering of emotional emptiness in eyes and features",

    # === ANGRY EXPRESSIONS (6) ===
    "{character}, {style}, intensely angry furious expression with deeply furrowed brow creating pronounced forehead wrinkles, narrowed blazing eyes with constricted pupils, flared nostrils showing heavy breathing, jaw clenched tight with visible tension, lips pressed thin or bared in snarl, dramatic harsh lighting with strong shadows amplifying aggression, tense confrontational atmosphere, highly detailed rendering of rage in every facial muscle",

    "{character}, {style}, frustrated irritated expression with pinched features showing annoyance, eyes rolling or narrowed in exasperation, tight lips pressed together holding back words, tense jaw and neck muscles visible, impatient agitated energy in features, direct unflattering lighting matching mood, stressed irritated atmosphere, detailed facial tension rendering",

    "{character}, {style}, cold controlled anger expression with icy piercing stare, rigidly controlled features masking fury underneath, clenched jaw with slight muscle twitch, lips in thin dangerous line, intimidating intensity in eyes, stark dramatic lighting creating harsh shadows, menacing dangerous atmosphere, detailed subtle signs of contained rage",

    "{character}, {style}, explosive outraged expression caught mid-shout with mouth wide open in yell, veins visible on forehead and neck from intensity, wild eyes with dilated pupils, face flushed red with fury, violent aggressive energy, chaotic dynamic lighting, overwhelming rage atmosphere, detailed capturing of extreme emotional state",

    "{character}, {style}, bitter resentful expression with sneering contemptuous curl to lip, cold hard eyes showing deep-seated anger, asymmetrical expression showing disgust, judgmental critical gaze, harsh unflattering lighting emphasizing negative emotion, toxic hostile atmosphere, detailed bitter expression lines",

    "{character}, {style}, defensive angry expression with jaw thrust forward in challenge, eyes narrowed in suspicion and accusation, protective aggressive posture reflected in face, cornered but fighting energy, confrontational lighting, tense standoff atmosphere, detailed defensive anger in features",

    # === SURPRISED EXPRESSIONS (5) ===
    "{character}, {style}, shocked surprised expression with eyes blown wide open showing whites around iris, raised eyebrows lifted high creating forehead wrinkles, jaw dropped open in genuine astonishment, frozen moment of unexpected discovery, dramatic moment lighting with strong contrast, high-impact surprise atmosphere, highly detailed rendering of sudden emotional reaction",

    "{character}, {style}, delighted pleasant surprise expression with eyes widening and mouth forming excited O shape, raised cheeks beginning to form smile, positive happy energy mixing with shock, uplifted surprised features, bright warm lighting matching positive nature, happy surprise atmosphere, detailed joyful shock rendering",

    "{character}, {style}, horrified shocked expression with wide terror-filled eyes, face drained of color showing pale complexion, mouth open in silent scream or gasp, features frozen in fear-tinged shock, dramatic horror lighting with harsh shadows, disturbing frightening atmosphere, detailed fearful surprise expression",

    "{character}, {style}, confused bewildered surprise expression with furrowed brow contradicting wide eyes, head tilted in incomprehension, mouth slightly open in questioning expression, puzzled disoriented appearance, uneven uncertain lighting matching confusion, disorienting perplexing atmosphere, detailed bewildered expression",

    "{character}, {style}, awestruck wonder surprise expression with wide sparkling eyes full of amazement, mouth open in breathless wonder, features lifted in reverent appreciation, magical discovery moment, ethereal glowing lighting suggesting wonder source, transcendent magical atmosphere, detailed rendering of profound amazement",

    # === SCARED/WORRIED EXPRESSIONS (5) ===
    "{character}, {style}, terrified frightened expression with dilated pupils in wide panicked eyes, face pale and drawn with fear, trembling visible in lip and chin, body language of wanting to flee visible in facial tension, eerie threatening lighting with harsh shadows, dangerous frightening atmosphere, highly detailed terror rendering",

    "{character}, {style}, anxious worried expression with brow knitted in concern, darting eyes showing nervous alertness, lip being bitten or trembling slightly, overall tense uncertain appearance, unsettling uneven lighting, anxiety-inducing tense atmosphere, detailed nervous expression features",

    "{character}, {style}, nervous apprehensive expression with tight controlled features masking underlying fear, swallowing visible in throat movement, eyes showing concealed worry, forced calm appearance with tells visible, clinical fluorescent lighting, uncomfortable waiting atmosphere, detailed subtle fear indicators",

    "{character}, {style}, paranoid suspicious expression with darting distrustful eyes, tense hypervigilant features, flinching reactive quality to expression, seeing threat everywhere in gaze, harsh surveillance-style lighting, unsafe threatening atmosphere, detailed paranoid expression rendering",

    "{character}, {style}, dread-filled expression with hollow eyes showing dark anticipation, resigned fearful features, pale complexion suggesting fear response, heavy emotional weight visible, ominous foreboding lighting, impending doom atmosphere, detailed expression of fearful anticipation",

    # === NEUTRAL/CALM EXPRESSIONS (4) ===
    "{character}, {style}, perfectly neutral expression with relaxed balanced features showing no strong emotion, even symmetrical arrangement of facial muscles, calm direct gaze without intensity, lips resting naturally neither smiling nor frowning, clean professional lighting with balanced key and fill, neutral studio atmosphere, highly detailed rendering of natural at-rest facial features",

    "{character}, {style}, serene peaceful expression with softened relaxed features radiating tranquility, gentle eyes with calm steady gaze, slight natural upturn to mouth suggesting contentment, overall harmonious balanced appearance, soft diffused lighting with gentle gradients, meditative calm atmosphere, detailed peaceful expression rendering",

    "{character}, {style}, composed collected expression with controlled measured features showing emotional regulation, steady unwavering eye contact, professional poised appearance, confident calm demeanor, professional portrait lighting, mature collected atmosphere, detailed composed expression features",

    "{character}, {style}, blank poker face expression with carefully controlled features revealing nothing, flat neutral eyes giving no emotional tells, still unmoving facial muscles, mysterious unreadable appearance, flat even lighting eliminating shadow tells, enigmatic mysterious atmosphere, detailed unreadable expression",

    # === THOUGHTFUL/CONTEMPLATIVE (5) ===
    "{character}, {style}, deeply thoughtful expression with focused inward-looking gaze, slight furrow in brow showing mental activity, chin perhaps resting on hand in classic thinking pose, processing considering appearance, warm intellectual lighting, scholarly contemplative atmosphere, highly detailed rendering of active thinking expression",

    "{character}, {style}, curious interested expression with slightly widened attentive eyes, tilted head suggesting active listening, open receptive facial features, engaged inquisitive appearance, bright welcoming lighting, discovering learning atmosphere, detailed curious expression rendering",

    "{character}, {style}, skeptical doubting expression with one eyebrow raised questioningly, slightly narrowed assessing eyes, pursed lips showing reservation, evaluating judging appearance, direct questioning lighting, challenging skeptical atmosphere, detailed doubtful expression features",

    "{character}, {style}, daydreaming distracted expression with unfocused distant gaze looking past camera, soft relaxed features with mind elsewhere, slight unconscious smile or neutral absent look, lost in thought appearance, dreamy soft lighting with hazy quality, imagination wandering atmosphere, detailed absent-minded expression",

    "{character}, {style}, concentrated focused expression with intense narrowed eyes on task, lips slightly parted in concentration, oblivious to surroundings with single-minded attention, deep focus appearance, practical task lighting, productive working atmosphere, detailed concentration expression rendering",

    # === COMPLEX/MIXED EXPRESSIONS (8) ===
    "{character}, {style}, bittersweet expression combining sad eyes with small smile, conflicting emotions visible simultaneously in different facial features, complex emotional state showing both happiness and sorrow, nuanced mixed feeling appearance, gentle conflicting lighting, emotionally complex atmosphere, detailed rendering of contradictory emotions",

    "{character}, {style}, nervous laughter expression with smile that doesn't reach worried eyes, tension visible around mouth despite grin, uncomfortable humor as coping mechanism, awkward conflicted appearance, uneven tense lighting, socially uncomfortable atmosphere, detailed nervous smile features",

    "{character}, {style}, tearful joy expression with crying while smiling widely, overwhelmed positive emotion causing tears, happy crinkled eyes with wet tears streaming, emotional overflow appearance, warm emotional lighting, overwhelming happiness atmosphere, detailed joyful crying rendering",

    "{character}, {style}, grudging respect expression with reluctant acknowledgment in eyes contradicting set jaw, fighting against expressing approval, conflicted impressed appearance, competing pride and resistance in features, complex dramatic lighting, reluctant admiration atmosphere, detailed conflicted expression",

    "{character}, {style}, forced smile expression with mouth curved up while eyes show different emotion, social mask not quite hiding true feeling underneath, polite facade with cracks visible, performative pleasant appearance, artificial social lighting, underlying tension atmosphere, detailed fake smile tells",

    "{character}, {style}, suspicious interest expression with narrowed eyes but engaged forward lean, distrust mixed with curiosity in features, wanting to know more despite reservations, cautious engagement appearance, shadowy intriguing lighting, dangerous curiosity atmosphere, detailed wary interest rendering",

    "{character}, {style}, nostalgic happiness expression with distant soft gaze remembering past, smile tinged with time passage sadness, memory playing behind eyes, wistful remembering appearance, warm vintage-toned lighting, memory lane atmosphere, detailed nostalgic expression features",

    "{character}, {style}, embarrassed pride expression with lowered eyes despite pleased smile, uncomfortable with attention while enjoying praise, modest discomfort with recognition, awkward pleasure appearance, warm spotlight lighting, proud embarrassment atmosphere, detailed humble pride rendering",

    # === EMOTIONAL INTENSITY VARIATIONS (10) ===
    "{character}, {style}, subtle micro smile with barely perceptible upturn of lip corners, gentle warmth visible mainly in softened eyes, sophisticated understated happiness, minimal refined expression, elegant soft lighting, subtle sophisticated atmosphere, highly detailed micro expression rendering",

    "{character}, {style}, mild annoyance expression with slight tightening around eyes, minor pursing of lips, controlled low-grade irritation, subtle discontent appearance, slightly harsh lighting, minor tension atmosphere, detailed subtle annoyance features",

    "{character}, {style}, gentle concern expression with soft worried eyes, slightly furrowed brow showing care, mild protective worry in features, caring attentive appearance, warm worried lighting, gentle care atmosphere, detailed subtle concern rendering",

    "{character}, {style}, quiet sadness expression with dimmed eyes and slight downturn to features, contained grief visible in subdued appearance, dignified sorrow showing, restrained emotional appearance, soft melancholic lighting, quiet grief atmosphere, detailed restrained sadness features",

    "{character}, {style}, extreme euphoria expression with every feature lifted in pure bliss, radiant glowing appearance, transcendent joy state, ultimate happiness showing, brilliant radiant lighting, peak joy atmosphere, detailed ecstatic expression rendering",

    "{character}, {style}, profound despair expression with completely devastated features, empty hollow look in eyes, soul-crushing sorrow visible, rock bottom emotional state, dark despairing lighting, hopeless atmosphere, detailed devastated expression features",

    "{character}, {style}, explosive rage expression with every muscle tensed in fury, face contorted with extreme anger, loss of control visible, maximum anger intensity, harsh violent lighting, dangerous explosive atmosphere, detailed extreme rage rendering",

    "{character}, {style}, absolute terror expression with maximum fear response in every feature, fight or flight intensity visible, extreme panic showing, primal fear state, horrific harsh lighting, nightmare terror atmosphere, detailed extreme fear features",

    "{character}, {style}, overwhelming love expression with complete adoration visible in softened features, eyes full of deep affection, profound emotional connection showing, deep love state, warm romantic lighting, intense love atmosphere, detailed profound adoration rendering",

    "{character}, {style}, pure wonder expression with childlike amazement in wide sparkling eyes, completely awed features, magical discovery moment showing, transcendent wonder state, magical ethereal lighting, enchanted wonder atmosphere, detailed pure amazement features",

    # === SITUATIONAL EXPRESSIONS (10) ===
    "{character}, {style}, receiving bad news expression with face falling as information processes, shock transitioning to grief, features crumpling with realization, devastating news moment, harsh clinical lighting, bad news delivery atmosphere, detailed reaction expression rendering",

    "{character}, {style}, keeping secret expression with knowing gleam in eyes, suppressed smile threatening to break through, barely contained excitement or knowledge, secret keeper appearance, conspiratorial dim lighting, hidden truth atmosphere, detailed secret keeping features",

    "{character}, {style}, caught lying expression with flickering eyes avoiding contact, micro expressions of guilt visible, uncomfortable squirming appearance, deception exposed moment, interrogation harsh lighting, guilty caught atmosphere, detailed caught expression features",

    "{character}, {style}, making decision expression with weighing consideration in focused eyes, processing multiple options visible, determination forming in features, choice moment showing, dramatic decisive lighting, crossroads atmosphere, detailed decision making features",

    "{character}, {style}, remembering trauma expression with distant haunted look, pain of memory visible in eyes, past surfacing in present features, traumatic recall moment, unsettling disturbing lighting, flashback atmosphere, detailed trauma memory features",

    "{character}, {style}, seeing loved one expression with face lighting up with recognition joy, features transforming with happiness, love and relief flooding expression, reunion moment showing, warm joyful lighting, happy reunion atmosphere, detailed recognition joy rendering",

    "{character}, {style}, steeling for difficulty expression with jaw setting in determination, eyes hardening with resolve, preparing mentally visible in features, gathering courage moment, dramatic strengthening lighting, facing challenge atmosphere, detailed preparation expression",

    "{character}, {style}, realizing mistake expression with dawning horror in widening eyes, features falling with comprehension, oh no moment visible, mistake recognition showing, harsh revealing lighting, terrible realization atmosphere, detailed dawning horror features",

    "{character}, {style}, giving up expression with defeat settling into features, hope draining from eyes, surrender to circumstances visible, capitulation moment, dim defeated lighting, giving up atmosphere, detailed defeated surrender rendering",

    "{character}, {style}, finding hope expression with light returning to eyes, features lifting with possibility, renewed energy visible in face, hope discovery moment, warming brightening lighting, hope emerging atmosphere, detailed hope finding features",
]


# ============================================================================
# ACTION TEMPLATES (143 templates) - DETAILED VERSION
# ============================================================================
# NOTE: Due to length, only showing first 50 action templates as examples
# Full implementation would include all 143 with similar detail level

ACTION_TEMPLATES = [
    # === BASKETBALL (6) ===
    "{character}, {style}, dynamic basketball dribbling action with athletic low stance and bent knees for stability, one hand controlling ball with fingertip touch while other arm extends for balance, intense focused gaze tracking invisible defenders, indoor gymnasium environment with polished wooden court floor visible, dramatic sports photography lighting with strong overhead lights creating dynamic shadows, energy and motion captured in ready-to-move pose, detailed athletic uniform with realistic sweat-wicking fabric texture, high quality sports action render",

    "{character}, {style}, basketball shooting pose at peak of jump shot with ball positioned at release point above head, perfect shooting form with elbow under ball and guide hand to side, legs extended from jump with toes pointed, focused eyes locked on basket target, indoor court environment with hoop visible in background, dramatic action lighting freezing peak moment, detailed basketball uniform with number visible, athletic footwear with accurate brand details",

    "{character}, {style}, powerful basketball dunk action with body elevated high above rim level, one arm fully extended reaching ball toward hoop, other arm out for balance, fierce determined expression with competitive intensity, arena environment with crowd blur in background, explosive action lighting with motion energy, detailed athletic jersey stretching with movement, dynamic body positioning showing power",

    "{character}, {style}, basketball defensive stance in ready position with knees bent and arms spread wide, intense focused eyes tracking ball handler, shuffling lateral movement suggested in foot position, determined competitive expression, gymnasium environment with court markings visible, intense game lighting with dramatic shadows, detailed defensive footwork position, athletic stance showing readiness",

    "{character}, {style}, basketball passing action with chest pass release position, arms extending forward pushing ball outward, stepping forward into pass with weight transfer, alert eyes on teammate target, fast-paced game environment, crisp action lighting capturing pass moment, detailed hand position on ball, athletic uniform showing team colors",

    "{character}, {style}, basketball rebounding action leaping high with arms stretched overhead, body fully extended reaching for ball at peak, competing for position in crowded paint, determined aggressive expression, intense game moment environment, powerful action lighting, detailed vertical leap athleticism, jersey riding up from jump effort",

    # === SOCCER (5) ===
    "{character}, {style}, powerful soccer kick action with striking leg fully extended connecting with ball, planted foot positioned beside ball for balance, body leaning into strike with intense focused expression, outdoor grass pitch environment with goal visible, dynamic sports action lighting, detailed soccer boot making contact, athletic uniform in team colors with grass stains",

    "{character}, {style}, soccer header action with body elevated and neck muscles tensed for impact, eyes open watching ball contact forehead, arms out for balance and timing, athletic jumping form, stadium environment with crowd background, dramatic action lighting capturing impact moment, detailed hair movement from header, intense competitive expression",

    "{character}, {style}, skillful soccer dribbling with ball close to feet in technical control, body feinting with deceptive movement, quick footwork suggested in leg positions, concentrated focused gaze on ball, grass pitch environment with defenders nearby, natural outdoor lighting, detailed ball control technique, athletic uniform showing movement",

    "{character}, {style}, soccer goalkeeper diving save with body fully horizontal in mid-air, arms stretched reaching for ball, determined face showing effort and focus, dramatic athletic stretch, goal mouth environment with net visible, action photography lighting freezing dive, detailed goalkeeper gloves and jersey, grass and dirt from pitch contact",

    "{character}, {style}, soccer celebration after scoring with arms raised in triumph, running joy with emotional release expression, teammate celebration environment, stadium atmosphere with crowd excitement, jubilant victory lighting, detailed jersey celebration, pure athletic joy captured",

    # === TENNIS (4) ===
    "{character}, {style}, powerful tennis forehand stroke at contact point with racket meeting ball, body rotated with full swing follow-through, athletic stance with bent knees and balanced weight, intense focused concentration on ball, tennis court environment with baseline visible, crisp action lighting, detailed racket strings and grip, athletic tennis outfit with accurate styling",

    "{character}, {style}, tennis backhand slice with elegant one-handed technique, racket cutting under ball with precision touch, graceful body rotation and footwork, concentrated focused expression, grass or hard court environment, classic tennis lighting, detailed racket positioning, pristine white tennis attire traditional styling",

    "{character}, {style}, explosive tennis serve at trophy position with arm raised and racket back, ball toss at perfect height, body coiled with potential energy, powerful athletic form, service line environment, dramatic serve lighting, detailed serving stance technique, athletic tennis wear stretching with motion",

    "{character}, {style}, quick tennis volley at net with compact punch stroke, fast reaction reflexes shown in ready position, aggressive net approach posture, alert anticipatory expression, net position on court, fast-paced action lighting, detailed volleying technique, athletic footwork positioning",

    # === SWIMMING (4) ===
    "{character}, {style}, freestyle swimming stroke with arm extended in pull phase, body streamlined and rotated for breathing, powerful kick visible underwater, focused determined expression, pool lane environment with lane lines visible, dramatic underwater-style lighting with caustic light patterns, detailed water interaction and splashing, athletic swimwear with competition styling",

    "{character}, {style}, competitive dive entry with body in streamlined position piercing water surface, arms pointed and legs together, minimal splash technique, focused pre-entry expression, diving pool environment, dramatic diving lighting, detailed body alignment, competition swimsuit details",

    "{character}, {style}, butterfly stroke at power phase with both arms pulling symmetrically, dolphin kick propulsion, breathing moment with head raised, intense athletic effort expression, competition pool environment, dynamic aquatic lighting, detailed water displacement, athletic swimming form",

    "{character}, {style}, backstroke swimming with arm recovery phase, body position on back rotating through water, steady kick rhythm, relaxed focused expression, pool ceiling environment visible, underwater lighting effects, detailed backstroke technique, streamlined swimming position",

    # === RUNNING/TRACK (5) ===
    "{character}, {style}, explosive sprint start from blocks with powerful driving phase, body at 45-degree forward angle, arms pumping with violent drive, intense determination expression, track starting blocks environment, dramatic start lighting, detailed sprinting mechanics, athletic track uniform and spikes",

    "{character}, {style}, mid-race sprinting at full speed with perfect running form, high knee lift and powerful arm drive, body upright in maximum velocity phase, fierce competitive focus, track lane environment, action sports lighting, detailed running stride mechanics, athletic track singlet and shorts",

    "{character}, {style}, hurdle clearance with lead leg extended over barrier, trail leg tucked tight to body, forward lean maintaining speed, focused concentrated expression, track hurdles environment, dramatic action lighting, detailed hurdle technique, athletic track uniform in motion",

    "{character}, {style}, long jump flight phase with body in hitchkick technique, arms cycling for balance, eyes focused on landing pit, athletic concentration expression, jumping pit environment, peak action lighting, detailed jump form, track and field uniform",

    "{character}, {style}, cross country running on trail with varied terrain, natural running stride adapting to ground, focused endurance expression, outdoor nature environment, natural daylight lighting, detailed trail running gear, athletic effort visible",

    # === GYMNASTICS (5) ===
    "{character}, {style}, perfect handstand hold with body in straight vertical line, fingers spread for balance, core engaged with pointed toes, focused concentrated expression, gymnastics mat environment, clean studio lighting, detailed body alignment, athletic leotard or gymnastics attire",

    "{character}, {style}, dynamic cartwheel mid-rotation with body in vertical plane, arms and legs in perfect wheel formation, controlled athletic movement, determined focused expression, gymnasium environment, action lighting capturing rotation, detailed gymnastic form, competition attire",

    "{character}, {style}, aerial somersault tucked in mid-flight with tight ball position, body rotating rapidly through air, athletic air awareness expression, high above mat environment, dramatic spotlight lighting, detailed tuck position, gymnastics competitive wear",

    "{character}, {style}, graceful split leap at peak height with legs in full split position, arms in elegant fifth position, joyful performance expression, stage or floor environment, performance lighting, detailed ballet-influenced form, performance leotard with sparkle details",

    "{character}, {style}, balance beam routine moment with one leg in arabesque, arms in graceful position for balance, poised confident expression, beam apparatus environment, competition lighting, detailed balance technique, elegant competition leotard",

    # === MARTIAL ARTS (6) ===
    "{character}, {style}, powerful high kick with leg extended above head height, supporting leg firmly planted, arms in guard position, fierce focused expression with kiai energy, dojo or training environment, dramatic martial arts lighting, detailed kick technique and form, traditional martial arts uniform",

    "{character}, {style}, boxing jab punch with lead arm fully extended, rear hand protecting chin, weight transfer forward, intense focused eyes on target, boxing ring or gym environment, dramatic fight lighting, detailed boxing stance and form, boxing gloves and athletic wear",

    "{character}, {style}, defensive martial arts block with forearm positioned to deflect strike, body turned to minimize target, ready counterattack stance, alert defensive expression, training hall environment, action lighting, detailed blocking technique, martial arts uniform",

    "{character}, {style}, roundhouse kick mid-rotation with hip fully turned, striking leg horizontal to ground, arms balanced for control, powerful intense expression, martial arts training environment, dynamic action lighting, detailed rotation mechanics, traditional training attire",

    "{character}, {style}, martial arts ready stance with feet shoulder-width apart, hands raised in guard position, weight centered and balanced, calm focused warrior expression, peaceful dojo environment, zen-like lighting, detailed fighting stance, traditional martial arts gi",

    "{character}, {style}, dynamic spinning back fist with body fully rotated, arm extending in strike, momentum of spin captured, determined attacking expression, training or competition environment, action photography lighting, detailed spinning technique, martial arts competition attire",

    # === VOLLEYBALL (4) ===
    "{character}, {style}, volleyball spike approach with powerful three-step run-up, arms swung back preparing for jump, eyes tracking set ball trajectory, intense focused attacking expression, indoor volleyball court environment with net visible, dramatic action lighting, detailed approach footwork mechanics, athletic volleyball uniform",

    "{character}, {style}, volleyball set action with fingers positioned overhead in perfect triangle formation, ball contact at forehead level, knees bent absorbing and redirecting momentum, concentrated focused expression, team game environment, bright gymnasium lighting, detailed setting hand technique, team volleyball attire",

    "{character}, {style}, volleyball dig defensive save with body low in platform position, arms together forming solid contact surface, quick lateral movement to ball, determined saving expression, court floor level environment, fast action lighting, detailed defensive technique, knee pads and athletic gear visible",

    "{character}, {style}, volleyball serve toss and approach with ball released to perfect height, body coiling for powerful jump serve, eyes tracking toss, confident attacking expression, service line environment, dramatic serve lighting, detailed serving preparation mechanics, athletic volleyball uniform",

    # === BASEBALL (4) ===
    "{character}, {style}, baseball batting swing at contact point with full hip rotation, bat meeting ball in perfect sweet spot, weight transferred to front foot, intense focused eyes on ball, home plate environment with diamond visible, dramatic action lighting freezing contact, detailed batting stance mechanics, full baseball uniform with helmet",

    "{character}, {style}, baseball pitching wind-up at balance point with knee raised high, ball hidden in glove, body coiled with potential energy, fierce competitive concentration, pitcher mound environment, dramatic sports lighting, detailed pitching mechanics, traditional baseball uniform",

    "{character}, {style}, baseball fielding catch with glove extended high for fly ball, body positioned under trajectory, focused tracking eyes, anticipatory catching expression, outfield environment with green grass, bright daylight stadium lighting, detailed catching form, baseball cap and fielding glove prominent",

    "{character}, {style}, baseball sliding into base with body horizontal in hook slide, hand reaching for bag, dust cloud rising from dirt, competitive intense expression, base path environment, action photography lighting, detailed sliding technique, dirt-stained uniform showing game effort",

    # === GOLF (3) ===
    "{character}, {style}, golf swing at moment of impact with club head striking ball, body in perfect rotated position, eyes fixed on contact point, controlled focused expression, fairway environment with manicured grass, bright outdoor lighting with shadows, detailed swing mechanics and form, traditional golf attire with polo and slacks",

    "{character}, {style}, golf putting stroke with pendulum motion, body perfectly still over ball, eyes reading line to hole, intense concentration expression, putting green environment with flag visible, soft consistent lighting, detailed putting stance and grip, classic golf attire",

    "{character}, {style}, golf address position with club head behind ball, knees flexed in athletic stance, hands gripping club properly, calm focused pre-shot expression, tee box environment, natural sunlight, detailed setup position mechanics, neat golf clothing with glove visible",

    # === SKIING (3) ===
    "{character}, {style}, alpine skiing carving turn with body angulated against slope, skis edging through snow creating spray, poles planted for rhythm, exhilarated focused expression, mountain slope environment with snow and sky, bright winter lighting, detailed skiing technique and form, full ski gear with goggles and helmet",

    "{character}, {style}, skiing aerial jump with body in tuck position mid-flight, skis parallel beneath body, arms forward for balance, thrilling determined expression, mountain terrain park environment, dramatic action lighting against snow, detailed jump form, professional skiing equipment",

    "{character}, {style}, cross-country skiing classic technique with diagonal stride, poles pushing powerfully, rhythmic gliding motion, endurance focused expression, winter trail environment, cold clear lighting, detailed nordic skiing mechanics, cross-country ski attire",

    # === CYCLING (3) ===
    "{character}, {style}, road cycling in aggressive aero position with back flat and hands on drops, powerful pedal stroke at full extension, eyes focused on road ahead, determined racing expression, open road environment, bright outdoor lighting, detailed cycling form and equipment, aerodynamic cycling kit and helmet",

    "{character}, {style}, mountain biking over obstacle with body lifted off saddle, bike tilted for balance, arms absorbing impact, focused technical expression, forest trail environment, dappled natural lighting, detailed mountain bike handling, protective MTB gear",

    "{character}, {style}, cycling sprint finish with explosive power in standing position, body rocking bike side to side, maximum effort pedaling, fierce competitive expression, finish line environment, dramatic sports lighting, detailed sprint mechanics, professional cycling equipment",

    # === SKATEBOARDING (3) ===
    "{character}, {style}, skateboard kickflip in mid-air with board rotating beneath feet, body elevated in pop, arms out for balance, stoked confident expression, urban skate park environment, dynamic action lighting, detailed skateboard trick mechanics, street skating attire",

    "{character}, {style}, skateboard grinding on rail with trucks locked on metal, body balanced in slide, arms adjusting for equilibrium, focused determined expression, urban street spot environment, gritty urban lighting, detailed grind position, casual skate clothing",

    "{character}, {style}, skateboard cruising push with back foot on ground propelling forward, relaxed flowing style, content cruising expression, sidewalk or street environment, warm afternoon lighting, detailed riding posture, casual streetwear style",

    # === YOGA (5) ===
    "{character}, {style}, warrior one yoga pose with front knee deeply bent at 90 degrees over ankle, back leg straight and strong pressing heel down, arms raised overhead with palms together reaching high, steady focused gaze forward in drishti, peaceful studio environment with natural light streaming through windows, calming soft yoga lighting with warm tones, detailed muscular engagement and alignment visible, fitted yoga attire showing clean body lines",

    "{character}, {style}, tree pose balance with standing leg rooted firmly into ground, raised foot pressed against inner thigh of standing leg, arms in graceful prayer position at heart center or extended overhead like branches, peaceful meditative expression with soft focused eyes, serene yoga space environment with plants and natural elements, soft diffused natural lighting, detailed balance technique and alignment, comfortable breathable yoga clothing",

    "{character}, {style}, downward facing dog pose with body forming inverted V shape, hands pressing firmly into mat with fingers spread, heels reaching toward floor stretching calves, head relaxed between arms with neck long, focused controlled breathing expression, yoga studio environment with wooden floors, even ambient lighting, detailed alignment showing spine length and hip position, yoga practice attire",

    "{character}, {style}, seated meditation pose in lotus position with legs crossed and feet on opposite thighs, hands resting on knees in mudra, spine tall and straight, eyes closed with peaceful serene expression, minimalist meditation space with cushion, soft ambient candlelight quality, detailed meditative stillness, loose comfortable meditation clothing",

    "{character}, {style}, cobra yoga pose with chest lifted from floor, hands pressing into mat beside ribs, hips and legs grounded, gentle backbend with head tilted back slightly, calm controlled expression, yoga mat on floor environment, warm natural lighting, detailed backbend form and engagement, fitted yoga attire",

    # === FITNESS/EXERCISE (8) ===
    "{character}, {style}, push-up at bottom position with chest near floor, body in perfect plank alignment, arms bent at sides, core engaged with straight line from head to heels, focused determined expression, gym floor or mat environment, practical workout lighting, detailed push-up form and muscle engagement, athletic workout attire",

    "{character}, {style}, squat exercise at bottom depth with thighs parallel to floor, knees tracking over toes, back straight with chest up, arms forward for balance, determined effort expression, gym or home workout environment, clear functional lighting, detailed squat mechanics and alignment, athletic training clothes",

    "{character}, {style}, plank hold with body forming rigid straight line, forearms or hands on floor, core tightly engaged, legs extended with weight on toes, focused endurance expression, exercise mat environment, practical lighting, detailed plank alignment and form, athletic workout gear",

    "{character}, {style}, deadlift at lockout standing tall with barbell at hip level, shoulders back and chest proud, hips fully extended, powerful satisfied expression, gym environment with equipment visible, strong gym lighting, detailed lifting form and posture, athletic training attire",

    "{character}, {style}, bicep curl mid-repetition with dumbbell raised halfway, elbow fixed at side, muscle visibly contracting, focused concentration on exercise, gym environment with mirrors, bright gym lighting showing muscle definition, detailed curl form and engagement, sleeveless workout attire",

    "{character}, {style}, jumping jack at peak with arms overhead and legs spread wide, full body extension in cardio movement, energetic active expression, open exercise space, bright energetic lighting, detailed jumping form and coordination, athletic cardio attire",

    "{character}, {style}, lunging exercise with front leg bent at 90 degrees, back knee lowering toward floor, torso upright with arms at sides or on hips, focused balanced expression, gym or outdoor environment, clear functional lighting, detailed lunge alignment, athletic training wear",

    "{character}, {style}, mountain climber exercise in plank position with one knee driven toward chest, dynamic running motion while stable, intense cardio effort expression, mat or floor environment, workout lighting, detailed dynamic form, athletic training clothes showing motion",

    # === DANCE (5) ===
    "{character}, {style}, ballet arabesque with one leg extended high behind body, supporting leg straight and strong, arms in graceful third position, elegant refined expression, ballet studio with barre and mirrors, classic dance lighting, detailed balletic line and form, traditional ballet attire with pointe shoes possible",

    "{character}, {style}, contemporary dance leap at peak height with legs in split, arms reaching expressively, body suspended in emotional movement, intense artistic expression, black box theater environment, dramatic stage lighting with colors, detailed contemporary dance form, flowing dance costume",

    "{character}, {style}, hip hop freeze pose in dramatic angular position, body creating geometric shapes, hands or head on ground for support, confident cool expression, urban dance studio environment, dynamic street dance lighting, detailed hip hop style and attitude, urban streetwear dance attire",

    "{character}, {style}, salsa partner dance frame with proper hold and connection, body in motion with hip movement suggested, engaging charming expression, dance floor environment with warm tones, latin club lighting with warm colors, detailed salsa posture and styling, latin dance attire",

    "{character}, {style}, breakdance windmill on back with legs spinning in circular motion, hands planted for support and rotation, dynamic athletic expression, dance floor or cardboard surface, dramatic hip hop lighting, detailed power move mechanics, baggy breakdance attire",

    # === MUSICAL PERFORMANCE (5) ===
    "{character}, {style}, guitar playing with fingers on fretboard forming chord, other hand strumming or picking strings, body connected to instrument, passionate musical expression, stage or studio environment, warm performance lighting with spotlight, detailed guitar playing technique, musician attire appropriate to genre",

    "{character}, {style}, piano playing with fingers dancing across keys, body slightly swaying with music, emotional connection to piece, absorbed musical expression, piano bench environment with instrument visible, soft performance lighting, detailed hand position on keys, concert or practice attire",

    "{character}, {style}, drums playing with sticks raised in motion between beats, body keeping rhythm, energetic groove expression, drum kit environment with cymbals visible, dynamic stage lighting with movement, detailed drumming technique and energy, casual musician attire",

    "{character}, {style}, singing into microphone with proper mic technique, body expressing through gesture, emotionally connected to lyrics, passionate performance expression, stage environment with audience suggestion, dramatic spotlight and stage lighting, detailed singing posture and engagement, performance outfit",

    "{character}, {style}, violin playing with bow drawing across strings, chin rest contact showing proper hold, body swaying with musical phrase, focused artistic expression, concert hall or studio environment, elegant performance lighting, detailed violin technique, formal concert attire",

    # === READING/WRITING (4) ===
    "{character}, {style}, reading book held comfortably in both hands with pages open at interesting angle, body in relaxed seated position in comfortable chair, completely absorbed expression with eyes scanning text, quiet cozy environment with bookshelves visible, warm lamp lighting creating reading atmosphere, detailed book and hand positioning showing engagement, casual comfortable loungewear clothing",

    "{character}, {style}, writing with pen in hand over paper or notebook, body leaning into task with focused concentration, thoughtful creative expression, desk or writing table environment, practical task lighting from lamp, detailed writing posture and pen grip, comfortable creative workspace attire",

    "{character}, {style}, typing on laptop with fingers hovering over keyboard, screen glow illuminating face, engaged working expression, modern workspace environment, mixed screen and ambient lighting, detailed typing posture and hand position, professional casual work attire",

    "{character}, {style}, studying with head resting on hand over open textbook, concentrated learning expression, surrounded by study materials, library or desk environment, practical study lighting, detailed studying posture showing effort, student casual attire",

    # === EATING/DRINKING (4) ===
    "{character}, {style}, eating meal with utensil raised bringing food toward mouth, enjoying delicious bite, satisfied pleased expression, dining table environment with meal visible, warm dining lighting, detailed eating posture and table manners, appropriate dining attire",

    "{character}, {style}, drinking from glass or cup held elegantly, liquid tilting toward lips, refreshed enjoying expression, cafe or dining environment, warm ambient lighting, detailed drinking gesture and glass hold, casual or semiformal attire",

    "{character}, {style}, coffee or tea enjoyment holding warm mug with both hands, steam rising from beverage, contented peaceful expression, cozy cafe or home environment, warm morning lighting, detailed mug holding gesture, comfortable casual clothing",

    "{character}, {style}, toast gesture raising wine glass with arm extended, celebratory cheers moment, happy social expression, celebration or dinner party environment, festive warm lighting, detailed toast posture and glass hold, elegant occasion attire",

    # === COOKING (3) ===
    "{character}, {style}, cooking at stove stirring pot with wooden spoon, monitoring dish progress, focused culinary concentration, kitchen environment with equipment visible, warm practical kitchen lighting, detailed cooking action and tools, casual home attire with apron",

    "{character}, {style}, chopping vegetables on cutting board with proper knife technique, focused precision in cuts, concentrated chef expression, kitchen prep area environment, clear task lighting, detailed knife skills and form, cooking attire with apron",

    "{character}, {style}, tasting dish from spoon with evaluating expression, culinary assessment moment, thoughtful critical expression, active kitchen environment, warm cooking lighting, detailed tasting gesture, chef or home cooking attire",

    # === SOCIAL GESTURES (10) ===
    "{character}, {style}, friendly wave gesture with raised hand and palm facing outward, fingers spread in welcoming greeting motion, warm genuine smile on face, outdoor public space or doorway environment, natural daylight lighting, detailed wave gesture and inviting expression, casual everyday clothing appropriate for social situation",

    "{character}, {style}, enthusiastic thumbs up with arm extended and thumb raised prominently, encouraging approval gesture, bright positive smile showing support, casual social environment, warm positive lighting, detailed hand gesture showing enthusiasm, casual friendly attire",

    "{character}, {style}, pointing gesture with arm extended and index finger indicating direction clearly, helpful guiding expression explaining something, contextual environment where direction matters, clear natural lighting, detailed pointing form and engaged expression, appropriate everyday clothing",

    "{character}, {style}, shrugging gesture with shoulders raised and palms turned upward, uncertain or questioning expression, casual conversational body language, everyday environment, natural ambient lighting, detailed shrug posture and expression, casual clothing",

    "{character}, {style}, clapping celebration with hands meeting in applause, genuine appreciation or congratulations, happy proud expression, event or performance environment, warm celebration lighting, detailed clapping motion and joyful expression, occasion-appropriate attire",

    "{character}, {style}, handshake greeting with firm professional grip, direct eye contact showing respect, confident professional expression, business or social introduction environment, professional lighting, detailed handshake form and engaged expression, professional or semiformal attire",

    "{character}, {style}, hugging embrace with arms wrapped around another in warm gesture, comforting supportive body language, caring emotional expression, intimate reunion or comfort environment, soft emotional lighting, detailed embrace showing genuine connection, casual everyday clothing",

    "{character}, {style}, thinking pose with hand on chin in contemplation, considering options or remembering, thoughtful processing expression, quiet environment for reflection, calm ambient lighting, detailed thinking gesture and expression, comfortable casual attire",

    "{character}, {style}, excited cheering with arms raised in victory or celebration, explosive happy energy, overjoyed ecstatic expression, celebratory event environment, bright festive lighting, detailed celebration gesture and emotion, fun casual or event attire",

    "{character}, {style}, comforting pat on shoulder reaching to console, supportive caring gesture, sympathetic kind expression, private supportive moment environment, soft warm lighting, detailed comforting touch and expression, everyday comfortable clothing",

    # === TEAM ACTIVITIES (6) ===
    "{character}, {style}, team huddle with arm around teammates in circle formation, unified group energy, determined team spirit expression, sports field or court environment, dramatic team lighting, detailed team bonding moment, matching team uniforms",

    "{character}, {style}, high five celebration with palm raised to meet teammate, shared victory moment, excited happy expression, sports or workplace environment, energetic positive lighting, detailed high five action, team or casual attire",

    "{character}, {style}, group discussion actively participating in conversation circle, engaged body language leaning in, interested contributing expression, meeting room or collaborative space, professional lighting, detailed discussion posture, appropriate professional attire",

    "{character}, {style}, collaborative work side by side with colleague on shared task, focused cooperative effort, engaged productive expression, workplace or project environment, practical working lighting, detailed collaborative posture, professional work attire",

    "{character}, {style}, team celebration group pose with arms raised or connected, unified joy moment, ecstatic celebratory expression, victory or achievement environment, triumphant lighting, detailed group celebration, team uniforms or celebration attire",

    "{character}, {style}, passing or tossing object to teammate with proper throwing form, cooperative action, focused helpful expression, team activity environment, action sports lighting, detailed passing technique, appropriate activity attire",

    # === DAILY ACTIVITIES (10) ===
    "{character}, {style}, waking up stretching in bed with arms raised overhead, morning energy release, sleepy awakening expression, bedroom environment with morning light, soft golden morning lighting through window, detailed stretching gesture, comfortable sleepwear or pajamas",

    "{character}, {style}, brushing teeth at bathroom sink with toothbrush in mouth, morning hygiene routine, sleepy routine expression, bathroom environment with mirror visible, bright bathroom lighting, detailed tooth brushing action, casual morning attire or pajamas",

    "{character}, {style}, getting dressed pulling shirt on or adjusting clothing, daily preparation moment, neutral focused expression, bedroom or closet environment, natural room lighting, detailed dressing action, partially dressed showing process",

    "{character}, {style}, walking casually in relaxed stride with natural arm swing, everyday locomotion, calm content expression, sidewalk or hallway environment, natural ambient lighting, detailed walking gait, casual everyday clothing",

    "{character}, {style}, opening door with hand on handle turning or pulling, transitional everyday action, neutral purposeful expression, doorway environment, practical lighting, detailed door opening gesture, everyday clothing",

    "{character}, {style}, sitting down onto chair or couch in process of lowering, transitional seated movement, relaxed comfortable expression, living space environment, warm home lighting, detailed sitting action, casual home attire",

    "{character}, {style}, looking at phone with device held at viewing angle, modern daily activity, engaged or curious expression, any casual environment, mixed screen and ambient lighting, detailed phone viewing posture, contemporary casual clothing",

    "{character}, {style}, carrying bag or backpack slung over shoulder, everyday transport, purposeful walking expression, urban or school environment, natural outdoor or indoor lighting, detailed carrying posture, appropriate casual attire",

    "{character}, {style}, cleaning or tidying with cloth or tool in hand, household task action, focused productive expression, home environment, practical daylight or room lighting, detailed cleaning action, casual work-appropriate home clothing",

    "{character}, {style}, greeting pet with affectionate bend toward animal, loving interaction moment, warm happy expression, home environment, soft warm lighting, detailed pet interaction posture, comfortable home attire",

    # === CLIMBING/ADVENTURE (4) ===
    "{character}, {style}, rock climbing with hands gripping holds and body pressed to wall, athletic climbing technique, focused determined expression, indoor climbing gym or outdoor rock environment, dramatic climbing lighting, detailed climbing position and form, proper climbing gear and attire",

    "{character}, {style}, hiking on trail with backpack and walking stride uphill, outdoor adventure activity, energized exploring expression, mountain or forest trail environment, natural outdoor lighting, detailed hiking posture and gear, appropriate outdoor hiking attire",

    "{character}, {style}, zip lining mid-flight holding harness with legs extended, adventure thrill activity, excited thrilled expression, aerial cable environment with scenery below, bright outdoor adventure lighting, detailed zip line position, adventure activity attire with safety gear",

    "{character}, {style}, surfing on wave in balanced riding stance, water sport action, stoked focused expression, ocean wave environment with spray, bright beach lighting, detailed surfing form and board position, wetsuit or surf attire",

    # === INTERACTION GESTURES (10) ===
    "{character}, {style}, offering hand to help with palm extended downward to assist, supportive helpful gesture, kind caring expression, situation requiring assistance environment, warm helpful lighting, detailed helping hand position, appropriate everyday attire",

    "{character}, {style}, receiving gift with hands cupped accepting present, grateful receiving moment, surprised happy expression, gift-giving occasion environment, warm celebration lighting, detailed receiving gesture, occasion-appropriate attire",

    "{character}, {style}, showing something holding object up to display, proud presenting gesture, enthusiastic sharing expression, contextual display environment, clear presentation lighting, detailed showing posture, appropriate casual attire",

    "{character}, {style}, asking question with raised hand or finger, inquiring seeking gesture, curious questioning expression, classroom or discussion environment, neutral practical lighting, detailed questioning pose, appropriate setting attire",

    "{character}, {style}, nodding agreement with head motion downward, affirmative response gesture, agreeable accepting expression, conversational environment, natural ambient lighting, detailed nodding moment, everyday clothing",

    "{character}, {style}, shaking head no with lateral head motion, disagreement refusal gesture, skeptical or denying expression, conversational environment, natural lighting, detailed head shake, casual everyday attire",

    "{character}, {style}, beckoning gesture with hand motion calling toward self, inviting welcoming action, encouraging friendly expression, entry or transitional environment, warm inviting lighting, detailed beckoning hand, welcoming casual attire",

    "{character}, {style}, pushing away gesture with arms extended palms out, rejecting defensive action, uncomfortable refusing expression, confrontational or protective environment, harsh or dramatic lighting, detailed pushing posture, appropriate everyday clothing",

    "{character}, {style}, leaning in to listen with body angled toward speaker, attentive engaged posture, interested curious expression, conversation or intimate environment, soft focused lighting, detailed listening pose, casual or professional attire",

    "{character}, {style}, stepping back in surprise with body weight shifting away, reactive retreat movement, startled surprised expression, unexpected encounter environment, sudden dramatic lighting, detailed retreat posture, everyday casual attire",

    # === EXTREME SPORTS (4) ===
    "{character}, {style}, parkour vault over obstacle with hands planted on surface pushing body over, legs tucked tight for clearance, focused determined expression, urban obstacle environment, dynamic action lighting, detailed parkour technique and form, athletic functional clothing",

    "{character}, {style}, snowboarding carving turn with board angled on edge cutting through powder, body low in athletic stance, arms out for balance, exhilarated thrilled expression, snowy mountain slope environment, bright winter sunlight, detailed snowboarding form, full winter sport gear with goggles",

    "{character}, {style}, bungee jumping mid-fall with arms spread wide in freefall, body stretched in controlled descent, mixture of fear and excitement expression, aerial environment with ground far below, dramatic action lighting, detailed freefall position, adventure activity attire with harness",

    "{character}, {style}, motocross bike jump with rider airborne over dirt mound, body position forward controlling bike, intense focused racing expression, outdoor dirt track environment, dusty action lighting, detailed riding technique, full protective motocross gear and helmet",

    # === WATER ACTIVITIES (3) ===
    "{character}, {style}, kayaking paddle stroke with double-bladed paddle pulling through water, body rotating for power, focused adventurous expression, river or lake environment with water visible, bright outdoor aquatic lighting, detailed paddling technique, water sport attire with life vest",

    "{character}, {style}, scuba diving swimming underwater with fins kicking slowly, arms relaxed at sides, calm peaceful exploring expression, underwater ocean environment with marine life, ethereal underwater lighting with light rays, detailed diving position, full scuba gear with mask and tank",

    "{character}, {style}, water polo treading water while holding ball overhead for throw, legs kicking underwater for elevation, competitive intense expression, pool environment with goals visible, bright aquatic lighting with water reflections, detailed water polo positioning, swim cap and competitive swimwear",

    # === CREATIVE ACTIVITIES (4) ===
    "{character}, {style}, painting on canvas with brush in hand adding detailed strokes, body leaning in with creative focus, absorbed artistic expression, art studio environment with supplies visible, natural north-facing studio lighting, detailed painting gesture and technique, casual artist attire with paint marks",

    "{character}, {style}, photography composing shot with camera raised to eye, body positioned for angle, concentrated creative expression, outdoor or studio shooting environment, mixed available and artificial lighting, detailed camera handling and photography stance, practical photographer attire",

    "{character}, {style}, sculpting clay or material with hands shaping form, fingers working on detailed area, deep artistic concentration expression, sculpture studio environment with tools, practical workshop lighting, detailed sculpting hand technique, work attire with clay marks",

    "{character}, {style}, drawing in sketchbook with pencil or pen making detailed marks, body hunched over work surface, absorbed creative expression, quiet workspace environment, practical task lighting, detailed drawing grip and technique, comfortable casual creative attire",

    # === COMMUNICATION/PRESENTATION (3) ===
    "{character}, {style}, public speaking at podium or stage with confident open gestures, body projecting to audience, engaging authoritative expression, lecture hall or conference environment, professional stage lighting with spotlight, detailed presentation posture and gesture, professional speaker attire",

    "{character}, {style}, teaching at whiteboard or screen pointing to content, body angled toward audience, enthusiastic explanatory expression, classroom or training room environment, even practical classroom lighting, detailed teaching gesture and stance, professional educator attire",

    "{character}, {style}, interview conversation seated with engaged listening posture, leaning forward with interest, attentive professional expression, office or studio interview environment, balanced interview lighting, detailed professional body language, formal interview appropriate attire",
]
