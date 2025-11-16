# docs/projects/turning_red – Supporting Character Files (Turning Red)

> This document bundles multiple character files. Split each `File:` section into its own markdown file under `docs/projects/turning_red/characters/`.

----------

## File: `character_ming_lee.md`

# Ming Lee — Character Reference (Turning Red)

## 1. Overview

Ming Lee is Meilin’s mother and the primary adult counterpart to Mei’s coming‑of‑age story. She is a loving but overprotective Chinese‑Canadian mother whose fear and unresolved trauma drive much of the film’s conflict. Emotionally she is a co‑protagonist: Mei’s arc cannot resolve without Ming facing her own past.

----------

## 2. Visual Design

### 2.1 Physical traits

-   Tall, slender adult woman in her 30s–40s.
    
-   Light skin tone; neat, smooth complexion.
    
-   Shoulder‑length dark hair, usually styled in a tidy wave with a part and soft volume.
    
-   Wears thin gold earrings and often a jade necklace or bracelet.
    

### 2.2 Signature outfits

-   **Green blazer look (most iconic):**
    
    -   Emerald green blazer or cardigan.
        
    -   Pale blouse underneath.
        
    -   Dark skirt or trousers.
        
    -   Low heels.
        
-   **Temple attire:**
    
    -   More traditional‑leaning red or maroon tops, sometimes with a qipao‑inspired collar.
        
-   **Casual / at‑home:**
    
    -   Knit sweaters, cardigans, and comfortable pants.
        

### 2.3 Expressions and body language

-   Upright posture; often stands with arms crossed or hands clasped.
    
-   Eyebrows tighten and lips press into a line when she worries.
    
-   When angry, eyes narrow, jaw clenches, and she looms over others; animators exaggerate her silhouette so she feels larger and more intimidating.
    

----------

## 3. Personality & Behavior

### 3.1 Core traits

-   Intensely protective and proud of Mei.
    
-   Highly organized, detail‑oriented, and image‑conscious.
    
-   Quick to panic and catastrophize if she thinks Mei is in danger or misbehaving.
    
-   Difficulty trusting other adults or peers with Mei; tends to intervene directly.
    

### 3.2 Flaws and vulnerabilities

-   Carries deep fear of repeating her own broken relationship with Grandma Wu.
    
-   Equates Mei’s obedience with love and respect, which makes her feel betrayed when Mei hides things.
    
-   Has trouble apologizing or showing vulnerability until the climax.
    

----------

## 4. Relationships

### 4.1 With Mei

-   At first, almost fused identity: Mei imitates Ming’s mannerisms and proudly parrots her values.
    
-   Conflict emerges when Mei’s private fantasies and panda form clash with Ming’s expectations.
    
-   Their relationship swings between tight co‑dependence and explosive confrontation.
    
-   Resolution: they meet as equals in the astral plane, recognizing each other’s fears and choosing a more balanced, less controlling relationship.
    

### 4.2 With Grandma Wu

-   Past panda incident led to a serious rift; Grandma’s scar is a visual reminder.
    
-   Ming still seeks her mother’s approval but is also resentful and afraid.
    
-   This unresolved tension shapes how Ming treats Mei.
    

### 4.3 With Jin

-   Jin acts as a calming counterweight.
    
-   Ming often takes the lead in decisions, with Jin supporting quietly; in later scenes he helps nudge her toward empathy.
    

----------

## 5. Key Scenes for Dataset Use

-   Dragging Mei to the Daisy Mart to confront Devon.
    
-   Overreacting to the sketchbook; driving angrily with Mei in the car.
    
-   Organizing and leading the first panda ritual.
    
-   Transforming into giant panda Ming at the SkyDome.
    
-   Astral plane scenes with younger Ming and Grandma Wu.
    
-   Quiet post‑climax conversation where she admits fear of losing Mei.
    

----------

## 6. Captioning & Tags

Suggested base tags:

```text
ming_lee, turning_red_style, chinese_canadian_mother,
adult woman, green blazer, neat dark hair,
protective expression, worried, strict, towering presence

```

Add relationship and scene tags: `with_mei_lee`, `giant_panda_form`, `ritual_scene`, `concert_kaiju_ming`, `astral_plane`.

----------

## 7. LoRA / Dataset Notes

-   For human‑Ming LoRA, focus on medium shots where her face and outfit are clear.
    
-   For giant panda Ming, treat as a separate concept (`ming_lee_red_panda`) with kaiju‑scale compositions, stadium backgrounds, and glowing red eyes.
    

----------

## File: `character_jin_lee.md`

# Jin Lee — Character Reference (Turning Red)

## 1. Overview

Jin Lee is Mei’s father, a quiet, supportive presence who balances Ming’s intensity. Though not as visually dominant as Ming or Mei, he is crucial to the film’s emotional resolution.

----------

## 2. Visual Design

-   Middle‑aged Chinese‑Canadian man with a slightly round build.
    
-   Short, neatly combed dark hair; wears rectangular glasses.
    
-   Often seen in soft sweaters, collared shirts, or aprons while cooking.
    
-   Color palette leans toward warm browns, creams, and muted greens.
    

----------

## 3. Personality & Behavior

-   Soft‑spoken, patient, and observant.
    
-   Uses humor and food to ease tension.
    
-   Rarely confrontational; prefers to listen and then quietly offer advice.
    
-   Provides some of the most emotionally grounded lines in the film.
    

----------

## 4. Key Relationship — With Mei

-   Jin sees Mei’s inner conflict more clearly than Ming does.
    
-   In a pivotal scene he shows Mei camcorder footage of her enjoying her panda form with friends, gently telling her that this side of herself is worth cherishing, not hiding.
    
-   This conversation nudges Mei toward choosing to keep the panda.
    

----------

## 5. Key Scenes for Dataset Use

-   Cooking in the family kitchen, serving food.
    
-   Filming Mei with a camcorder at home or the temple.
    
-   Quietly watching Ming and Mei argue, then offering a small reaction shot.
    
-   Sitting with Mei during his advice scene.
    

----------

## 6. Captioning & Tags

```text
jin_lee, turning_red_style, chinese_canadian_father,
soft_spoken, glasses, sweater_vest, gentle_expression,
serving_food, camcorder

```

----------

## 7. LoRA Notes

Jin is useful mainly for **family ensemble** training and father‑figure body language. He can also anchor scenes where you want a calmer adult presence to contrast with Mei and Ming’s extremes.

----------

## File: `character_grandma_wu_and_aunties.md`

# Grandma Wu & Aunties — Character Reference (Turning Red)

## 1. Overview

Grandma Wu and Ming’s sisters form a formidable matriarchal unit. They represent the older generation of Lee women who have all carried the panda and made their own compromises with it.

----------

## 2. Visual Design

-   Travel in a cluster, each aunt with a distinct silhouette and color palette.
    
-   Grandma Wu: small but commanding presence, short gray hair, traditional clothing, and a visible scar on her eyebrow or temple.
    
-   Aunties: varied heights and builds; outfits mix traditional Chinese patterns with contemporary fashion.
    

----------

## 3. Personality & Behavior

-   Grandma Wu: stern, authoritative, but quietly loving; speaks with weight and finality.
    
-   Aunties: gossiping, supportive, expressive; provide comic relief while still upholding tradition.
    

----------

## 4. Key Scenes

-   Airport arrival in a tight, intimidating group shot.
    
-   Preparing for the ritual, discussing Ming’s past.
    
-   Shocked reactions during Mei’s defiance.
    
-   Breaking their talismans and transforming into pandas at the concert.
    
-   Supporting Ming during the astral ritual.
    

----------

## 5. Captioning & Tags

```text
grandma_wu, lee_aunties, turning_red_style,
elder_chinese_women, traditional_clothing,
ritual_scene, group_shot

```

You can also tag individual aunties if desired (e.g., `auntie_chen`, `auntie_ping`, etc.) for finer‑grained datasets.

----------

## File: `character_miriam_mendelsohn.md`

# Miriam Mendelsohn — Character Reference (Turning Red)

## 1. Overview

Miriam is Mei’s best friend and emotional anchor. She embodies chill, supportive energy and is often the first to check in when Mei is struggling.

----------

## 2. Visual Design

-   Tall, lanky teenage girl.
    
-   Light skin with freckles; braces on her teeth.
    
-   Wavy light‑brown hair, usually under a beanie.
    
-   Skater / tomboy style: layered shirts, baggy pants, sneakers, and bracelets.
    

----------

## 3. Personality & Behavior

-   Friendly, empathetic, quick to make jokes.
    
-   Often the peacekeeper in the group.
    
-   Plays the role of “band’s bass player” type — solid, supportive, not always in the spotlight.
    

----------

## 4. Key Scenes

-   Hyping Mei up at school, drumming on the desk.
    
-   Reacting with delight, not fear, to panda‑Mei.
    
-   Being visibly hurt when Mei fails to stand up for her friends.
    
-   Forgiving Mei and singing along at the concert.
    

----------

## 5. Captioning & Tags

```text
miriam_mendelsohn, turning_red_style, tall_teen_girl,
freckles, braces, beanie, skater_outfit,
with_mei_lee, best_friend

```

----------

## File: `character_priya_mangal.md`

# Priya Mangal — Character Reference (Turning Red)

## 1. Overview

Priya is the deadpan member of Mei’s friend group, bringing mellow, goth‑leaning energy to the quartet.

----------

## 2. Visual Design

-   Brown skin; Indian heritage.
    
-   Long dark hair with straight bangs.
    
-   Wears glasses.
    
-   Clothing tends toward yellows, browns, and blacks, with subtle goth/punk accessories.
    

----------

## 3. Personality & Behavior

-   Speaks in a low, monotone voice; rarely shows big facial expressions.
    
-   Enjoys dark or morbid media (vampire romance novels, goth culture).
    
-   When she does get excited (dancing, cheering), it’s visually striking because it contrasts with her usual calm demeanor.
    

----------

## 4. Key Scenes

-   Reading on the bleachers or in the background.
    
-   Dancing intensely yet expressionlessly at Tyler’s party.
    
-   Giving dry one‑liners in response to Mei’s drama.
    

----------

## 5. Captioning & Tags

```text
priya_mangal, turning_red_style, teen_girl,
indian_descent, glasses, long_dark_hair_with_bangs,
monotone_expression, goth_aesthetic

```

----------

## File: `character_abby_park.md`

# Abby Park — Character Reference (Turning Red)

## 1. Overview

Abby is the smallest but loudest member of Mei’s friend group, embodying chaotic, explosive enthusiasm.

----------

## 2. Visual Design

-   Short, compact build.
    
-   Round face with thick eyebrows.
    
-   Wears purple overalls, a striped shirt, and sometimes a headband.
    
-   Korean heritage.
    

----------

## 3. Personality & Behavior

-   Loud, intense, quick to anger on Mei’s behalf.
    
-   Also extremely affectionate, especially toward the red panda.
    
-   Expressive body language: clenched fists, stomping feet, sudden lunges into hugs or shouts.
    

----------

## 4. Key Scenes

-   Screaming at classmates about environmental issues.
    
-   Throwing herself onto panda‑Mei in a crushing hug.
    
-   Threatening Tyler when he bullies Mei.
    
-   Jumping and flailing in excitement at 4★Town.
    

----------

## 5. Captioning & Tags

```text
abby_park, turning_red_style, short_teen_girl,
purple_overalls, thick_eyebrows, explosive_expression,
with_mei_lee, hyperactive

```
## File: `character_mr_gao.md`

# Mr. Gao — Character Reference (Turning Red)

## 1. Overview

Mr. Gao is an older family friend and ritual specialist who guides the Lee women through the panda sealing ceremony.

----------

## 2. Visual Design

-   Elderly Chinese man with a kind face.
    
-   Wears traditional‑style clothing, often in muted blues or grays.
    
-   Slightly stooped posture, hands often clasped behind his back.
    

----------

## 3. Personality & Behavior

-   Calm, humorous, unfazed by the supernatural.
    
-   Treats the panda curse as both sacred and routine, making jokes while still respecting the ritual.
    

----------

## 4. Key Scenes

-   Performing the initial ritual on Mei.
    
-   Leading the combined ritual at the concert.
    
-   Giving wry commentary about the panda and family history.
    

----------

## 5. Captioning & Tags

```text
mr_gao, turning_red_style, elderly_chinese_man,
ritual_leader, calm_expression, traditional_clothing

```

----------

## File: `character_devon.md`

# Devon — Character Reference (Turning Red)

## 1. Overview

Devon is a teen convenience‑store clerk who becomes the focus of Mei’s early crush fantasies.

----------

## 2. Visual Design

-   Teen boy with a slightly lanky build.
    
-   Wears Daisy Mart uniform: vest or polo shirt with store logo, name tag, and cap.
    
-   Often framed behind the counter or near fridge aisles.
    

----------

## 3. Personality & Behavior

-   Chill, slightly oblivious; not central to the main plot.
    
-   Functions mostly as an object of Mei’s projected fantasies.
    

----------

## 4. Key Scenes

-   Mei secretly sketching idealized Devons in her notebook.
    
-   The confrontation where Ming shows him the drawings, leaving him confused.
    
-   Brief later cameos during neighborhood scenes.
    

----------

## 5. Captioning & Tags

```text
devon_daisy_mart, turning_red_style, teen_clerk,
convenience_store_uniform, crush_target

```

----------

## File: `character_stacy_and_classmates.md`

# Stacy & Classmates — Character Reference (Turning Red)

## 1. Overview

Stacy and the other classmates serve mainly as a chorus of reactions: they amplify school gossip, spread panda rumors, and form the cheering crowds.

----------

## 2. Visual Notes

-   Mixed group of middle‑school students with diverse ethnicities and body types.
    
-   Stacy: ponytail, casual school clothes, expressive face; often at the center of rumor‑spreading moments.
    

----------

## 3. Usage in Datasets

-   Useful for crowd shots, school hallway backgrounds, and reaction faces to panda‑Mei.
    
-   Can be tagged generically (`classmates`, `school_crowd`, `stacy`) unless you want individual character LoRAs.
    

----------

## 4. Captioning & Tags

```text
stacy, turning_red_style, school_girl,
classmates_group, school_hallway_crowd,
reaction_shot

```

----------

## File: `character_4town.md`

# 4★Town — Group Character Reference (Turning Red)

## 1. Overview

4★Town is the in‑universe boy band adored by Mei and her friends. The group includes Robaire, Jesse, Aaron Z., Aaron T., and Tae Young.

----------

## 2. Visual Design

-   Each member has distinct hair, outfit, and color palette but all fit the early‑2000s boy band aesthetic: layered outfits, coordinated colors, chunky sneakers, earrings, and jewelry.
    
-   Stage design includes light rigs, smoke, confetti, and animated background screens.
    

----------

## 3. Personality & Roles (high‑level)

-   **Robaire:** Charismatic leader, sleek hairstyle, often in center.
    
-   **Jesse:** Older, more mature vibe, slightly dad‑like.
    
-   **Aaron Z. & Aaron T.:** Energetic dancers, playful personalities.
    
-   **Tae Young:** Soft, kind, animal‑loving member.
    

----------

## 4. Key Scenes

-   Promotional images and posters in Mei’s room.
    
-   SkyDome concert performance, including synchronized choreography.
    
-   Helping anchor the ritual chant with their music during the climax.
    

----------

## 5. Captioning & Tags

```text
4town, boy_band, turning_red_style,
stage_concert, synchronized_dance,
rob


```
Tyler
## File: `character_tyler_nguyen_baker.md`

# Tyler Nguyen‑Baker — Character Reference (Turning Red)

## 1. Overview

Tyler begins as Mei’s obnoxious classmate but eventually becomes a genuine friend and fellow 4★Town fan.

----------

## 2. Visual Design

-   Teen boy with tan skin and messy hair.
    
-   Wears brightly colored sports jerseys and baggy shorts.
    
-   Often seen with a smug grin or mocking expression.
    

----------

## 3. Personality & Behavior

-   Loud, teasing, and attention‑seeking.
    
-   Uses Mei as a target for jokes but is secretly insecure.
    
-   His 4★Town fandom reveals a softer, more earnest side.
    

----------

## 4. Key Scenes

-   Teasing Mei after the Devon incident.
    
-   Hosting a birthday party and hiring panda‑Mei as an attraction.
    
-   Being attacked when Mei loses control.
    
-   Appearing at the concert in full 4★Town fan gear.
    
-   Hanging out with the group in the epilogue.
    

----------

## 5. Captioning & Tags

```text
tyler_nguyen_baker, turning_red_style, teen_boy,
sports_jersey, smug_expression, bully_to_friend,
4town_fan

```





