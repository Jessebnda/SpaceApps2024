
from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline
from sentence_transformers import SentenceTransformer, util
import torch

# Load models
model_name = "distilbert/distilbert-base-cased-distilled-squad"
nlp = pipeline('question-answering', model=model_name, tokenizer=model_name)
embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# Predefined context with corresponding descriptions
contexts = {
    """
    Mars is one of the most explored bodies in our solar system, and it's the only planet where we've sent rovers to roam the alien landscape. NASA missions have found lots of evidence that Mars was much wetter and warmer, with a thicker atmosphere, billions of years ago.

    Mars was named by the Romans for their god of war because its reddish color was reminiscent of blood. The Egyptians called it "Her Desher," meaning "the red one."

    Mars has two small moons, Phobos and Deimos, that may be captured asteroids. They're potato-shaped because they have too little mass for gravity to make them spherical.

    The moons get their names from the horses that pulled the chariot of the Greek god of war, Ares.

    Even today, it is frequently called the "Red Planet" because iron minerals in the Martian dirt oxidize, or rust, causing the surface to look red.

    With a radius of 2,106 miles (3,390 kilometers), Mars is about half the size of Earth. If Earth were the size of a nickel, Mars would be about as big as a raspberry.

    From an average distance of 142 million miles (228 million kilometers), Mars is 1.5 astronomical units away from the Sun. One astronomical unit (abbreviated as AU), is the distance from the Sun to Earth. From this distance, it takes sunlight 13 minutes to travel from the Sun to Mars.

    Mars has a dense core at its center between 930 and 1,300 miles (1,500 to 2,100 kilometers) in radius. It's made of iron, nickel, and sulfur. Surrounding the core is a rocky mantle between 770 and 1,170 miles (1,240 to 1,880 kilometers) thick, and above that, a crust made of iron, magnesium, aluminum, calcium, and potassium. This crust is between 6 and 30 miles (10 to 50 kilometers) deep.

    Mars has a thin atmosphere made up mostly of carbon dioxide, nitrogen, and argon gases. To our eyes, the sky would be hazy and red because of suspended dust instead of the familiar blue tint we see on Earth. Mars' sparse atmosphere doesn't offer much protection from impacts by such objects as meteorites, asteroids, and comets.

    The temperature on Mars can be as high as 70 degrees Fahrenheit (20 degrees Celsius) or as low as about -225 degrees Fahrenheit (-153 degrees Celsius). And because the atmosphere is so thin, heat from the Sun easily escapes this planet. If you were to stand on the surface of Mars on the equator at noon, it would feel like spring at your feet (75 degrees Fahrenheit or 24 degrees Celsius) and winter at your head (32 degrees Fahrenheit or 0 degrees Celsius).
    """
     
        
    "Why is Mars one of the most explored bodies in our solar system?": "Mars is one of the most explored bodies in our solar system, and it's the only planet where we've sent rovers to roam the alien landscape.",
    "What kind of evidence has NASA found about Mars' past?": "NASA missions have found lots of evidence that Mars was much wetter and warmer, with a thicker atmosphere, billions of years ago.",
    "Why did the Romans name Mars after their god of war?": "Mars was named by the Romans for their god of war because its reddish color was reminiscent of blood.",
    "What did the Egyptians call Mars?": "The Egyptians called Mars 'Her Desher,' meaning 'the red one.'",
    "How many moons does Mars have?": "Mars has two small moons, Phobos and Deimos.",
    "Why are Mars' moons not spherical in shape?": "Mars' moons are potato-shaped because they have too little mass for gravity to make them spherical.",
    "What might be the origin of Mars' moons?": "Mars' moons may be captured asteroids.",
    "What are Mars' moons named after?": "The moons are named after the horses that pulled the chariot of the Greek god of war, Ares.",
    "Why is Mars frequently called the 'Red Planet'?": "Mars is frequently called the 'Red Planet' because iron minerals in the Martian dirt oxidize, or rust, causing the surface to look red.",
    "What is the radius of Mars compared to Earth's?": "With a radius of 2,106 miles (3,390 kilometers), Mars is about half the size of Earth.",
    "How would you compare the size of Mars to an everyday object?": "If Earth were the size of a nickel, Mars would be about as big as a raspberry.",
    "What is Mars' average distance from the Sun?": "From an average distance of 142 million miles (228 million kilometers), Mars is 1.5 astronomical units away from the Sun.",
    "How long does sunlight take to travel from the Sun to Mars?": "From this distance, it takes sunlight 13 minutes to travel from the Sun to Mars.",
    "What materials make up Mars' core?": "Mars has a dense core made of iron, nickel, and sulfur.",
    "How thick is Mars' mantle?": "Mars has a rocky mantle between 770 and 1,170 miles (1,240 to 1,880 kilometers) thick.",
    "What materials are found in Mars' crust?": "Mars' crust is made of iron, magnesium, aluminum, calcium, and potassium.",
    "How deep is Mars' crust?": "Mars' crust is between 6 and 30 miles (10 to 50 kilometers) deep.",
    "What gases make up Mars' thin atmosphere?": "Mars has a thin atmosphere made up mostly of carbon dioxide, nitrogen, and argon gases.",
    "What would the sky look like on Mars to the human eye?": "To our eyes, the sky on Mars would be hazy and red because of suspended dust.",
    "Why doesn't Mars' atmosphere offer much protection from impacts?": "Mars' sparse atmosphere doesn't offer much protection from impacts by meteorites, asteroids, and comets.",
    "What are the temperature extremes on Mars?": "The temperature on Mars can be as high as 70 degrees Fahrenheit (20 degrees Celsius) or as low as about -225 degrees Fahrenheit (-153 degrees Celsius).",
    "Why does heat easily escape from Mars?": "Because the atmosphere is so thin, heat from the Sun easily escapes Mars.",
    "What temperature differences would you feel standing on Mars at the equator?": "If you stood on the surface of Mars on the equator at noon, it would feel like spring at your feet (75°F or 24°C) and winter at your head (32°F or 0°C).",
    "How does Mars compare in size to Earth?": "Mars is about half the size of Earth.",
    "What are the characteristics of Mars' core?": "Mars' core has a radius between 930 and 1,300 miles (1,500 to 2,100 kilometers) and is made of iron, nickel, and sulfur.",
    "How does the Martian sky differ from Earth's?": "The Martian sky would appear hazy and red due to suspended dust, unlike the blue sky on Earth.",
    "What causes the surface of Mars to appear red?": "The surface of Mars looks red because iron minerals in the Martian dirt oxidize, or rust.",
    "How does the size of Phobos and Deimos affect their shape?": "Phobos and Deimos are too small for gravity to make them spherical, so they are potato-shaped.",
    "How does Mars' atmosphere compare to Earth's?": "Mars' atmosphere is much thinner than Earth's and made mostly of carbon dioxide, nitrogen, and argon.",
    "How do seasonal temperature variations on Mars differ from Earth's?": "Mars experiences more extreme temperature variations due to its thin atmosphere, which allows heat to escape easily.",
    "What historical evidence shows that Mars was once warmer?": "NASA missions have found evidence that Mars was much wetter and warmer billions of years ago.",
    "What role does Mars' atmosphere play in surface impacts?": "Mars' thin atmosphere offers little protection against impacts from meteorites and asteroids.",
    "Why do Phobos and Deimos have irregular shapes?": "Phobos and Deimos are irregularly shaped because they may be captured asteroids with insufficient mass to form spheres.",
    "How does the mass of Mars compare to Earth’s?": "Mars is less massive than Earth, contributing to its weaker gravity and inability to retain a thick atmosphere.",
    "What is the likely origin of Mars' moons?": "Mars' moons may have originated as captured asteroids from the asteroid belt.",
    "What minerals are found in the Martian crust?": "Mars' crust contains iron, magnesium, aluminum, calcium, and potassium.",
    "Why is Mars' atmosphere unable to retain much heat?": "Mars' thin atmosphere allows heat from the Sun to escape easily, leading to rapid cooling.",
    "Why is Mars considered the only planet with rovers on its surface?": "Mars is the only planet where we've sent rovers to roam the alien landscape.",
    "What have NASA missions discovered about Mars billions of years ago?": "NASA missions have found lots of evidence that Mars was much wetter and warmer, with a thicker atmosphere, billions of years ago.",
    "What is the significance of Mars being named after the Roman god of war?": "Mars was named by the Romans for their god of war because its reddish color was reminiscent of blood.",
    "What does the Egyptian name 'Her Desher' mean and to which planet does it refer?": "The Egyptians called Mars 'Her Desher,' meaning 'the red one.'",
    "What are the names of Mars' two moons and what is their possible origin?": "Mars has two small moons, Phobos and Deimos, that may be captured asteroids.",
    "Why do Phobos and Deimos have irregular, potato-like shapes?": "They're potato-shaped because they have too little mass for gravity to make them spherical.",
    "From which mythological figures do Mars' moons derive their names?": "The moons are named after the horses that pulled the chariot of the Greek god of war, Ares.",
    "What causes the reddish appearance of Mars' surface?": "Iron minerals in the Martian dirt oxidize, or rust, causing the surface to look red.",
    "How does the radius of Mars compare to that of Earth?": "With a radius of 2,106 miles (3,390 kilometers), Mars is about half the size of Earth.",
    "If Earth were the size of a nickel, how large would Mars be?": "If Earth were the size of a nickel, Mars would be about as big as a raspberry.",
    "What is Mars' average distance from the Sun in astronomical units (AU)?": "From an average distance of 142 million miles (228 million kilometers), Mars is 1.5 astronomical units away from the Sun.",
    "How long does it take for sunlight to travel from the Sun to Mars?": "From this distance, it takes sunlight 13 minutes to travel from the Sun to Mars.",
    "What elements make up the core of Mars?": "Mars has a dense core made of iron, nickel, and sulfur.",
    "How thick is the rocky mantle of Mars?": "Mars has a rocky mantle between 770 and 1,170 miles (1,240 to 1,880 kilometers) thick.",
    "Which elements are found in the crust of Mars?": "Mars' crust is made of iron, magnesium, aluminum, calcium, and potassium.",
    "What is the depth range of Mars' crust?": "Mars' crust is between 6 and 30 miles (10 to 50 kilometers) deep.",
    "What gases primarily compose Mars' atmosphere?": "Mars has a thin atmosphere made up mostly of carbon dioxide, nitrogen, and argon gases.",
    "How would the Martian sky appear to human observers?": "To our eyes, the sky on Mars would be hazy and red because of suspended dust.",
    "Why doesn't Mars' atmosphere provide much protection from space impacts?": "Mars' sparse atmosphere doesn't offer much protection from impacts by meteorites, asteroids, and comets.",
    "What are the maximum and minimum temperatures recorded on Mars?": "The temperature on Mars can be as high as 70 degrees Fahrenheit (20 degrees Celsius) or as low as about -225 degrees Fahrenheit (-153 degrees Celsius).",
    "Why does heat escape easily from Mars?": "Because the atmosphere is so thin, heat from the Sun easily escapes Mars.",
    "What temperature variations would one experience standing on Mars at the equator at noon?": "If you stood on the surface of Mars on the equator at noon, it would feel like spring at your feet (75°F or 24°C) and winter at your head (32°F or 0°C).",
    "How does the mass of Mars affect its ability to retain an atmosphere?": "Mars is less massive than Earth, contributing to its weaker gravity and inability to retain a thick atmosphere.",
    "What is the composition of Mars' mantle?": "Mars has a rocky mantle composed of iron, magnesium, aluminum, calcium, and potassium.",
    "How does the thickness of Mars' crust compare to its mantle?": "Mars' crust is much thinner, between 6 and 30 miles (10 to 50 kilometers) deep, compared to the mantle's 770 to 1,170 miles (1,240 to 1,880 kilometers).",
    "What evidence suggests that Mars once had a thicker atmosphere?": "NASA missions have found evidence that Mars was much wetter and warmer, with a thicker atmosphere, billions of years ago.",
    "How do the temperature extremes on Mars affect its climate?": "The extreme temperatures, ranging from 70°F (20°C) to -225°F (-153°C), contribute to Mars' harsh and variable climate.",
    "What role do iron, magnesium, aluminum, calcium, and potassium play in Mars' crust?": "These elements make up the composition of Mars' crust, contributing to its geological structure.",
    "Why might Mars' moons have been captured asteroids?": "Their irregular, potato-like shapes and small masses suggest they may have been captured by Mars' gravity.",
    "How does the thin atmosphere of Mars influence its surface conditions?": "The thin atmosphere leads to hazy, red skies and offers little protection from space debris impacts.",
    "What geological features on Mars indicate past water activity?": "NASA missions have found evidence of Mars being much wetter and warmer in the past, suggesting features like dried-up riverbeds and mineral deposits.",
    "How does the size of Mars influence its geological activity compared to Earth?": "Mars' smaller size and lower mass result in less geological activity and a thinner atmosphere.",
    "What is the significance of the thin atmosphere for potential human exploration of Mars?": "A thin atmosphere means less protection from radiation and extreme temperatures, posing challenges for human habitation.",
    "How do the names of Mars' moons reflect their mythological origins?": "Phobos and Deimos are named after the horses that pulled Ares' chariot, linking them to the Greek god of war.",
    "What challenges do the temperature extremes on Mars present for rovers and equipment?": "Extreme cold can affect battery life and the functionality of mechanical parts, requiring robust engineering solutions.",
    "How does the distance of Mars from the Sun affect its climate and potential for life?": "Being 1.5 AU from the Sun, Mars receives less solar energy, contributing to its colder climate and challenging conditions for life.",
    "What geological processes have shaped the surface of Mars?": "Processes like volcanic activity, erosion by wind and water, and impacts from meteorites have shaped Mars' surface.",
    "Why is Mars a focus for studying the potential for past life in the solar system?": "Evidence of past water and a thicker atmosphere suggests Mars may have had conditions suitable for life billions of years ago.",
    "How does the size comparison between Earth and Mars help in understanding their differences?": "Mars being about half the size of Earth highlights differences in gravity, atmosphere retention, and geological activity.",
    "What is the importance of studying Mars' moons Phobos and Deimos?": "Studying Mars' moons can provide insights into the planet's gravitational influence and the history of the solar system's small bodies.",
    "How does the presence of iron oxide on Mars' surface influence its appearance?": "Iron oxide, or rust, gives Mars its characteristic red color, making it visually distinctive.",
    "What are the implications of Mars' thin atmosphere for its ability to support liquid water today?": "A thin atmosphere results in low atmospheric pressure and temperatures that make liquid water unstable on the surface.",
    "How do the atmospheric conditions on Mars compare to those on Earth in terms of habitability?": "Mars' thin atmosphere, composed mainly of carbon dioxide, offers little protection and makes it less habitable compared to Earth's thick, oxygen-rich atmosphere.",
    "What technological advancements have enabled rovers to operate on Mars?": "Advancements in robotics, autonomous navigation, energy-efficient power systems, and durable materials have enabled rovers to operate in Mars' harsh environment.",
    "Why is understanding Mars' geological history important for planetary science?": "It helps scientists learn about planetary formation, climate evolution, and the potential for life beyond Earth.",
    "How does Mars' axial tilt influence its seasons compared to Earth's?": "Mars has an axial tilt similar to Earth's, causing it to experience seasons, but the thin atmosphere leads to more extreme temperature variations.",
    "What are the primary challenges in sending humans to Mars based on its environment?": "Challenges include extreme temperatures, low atmospheric pressure, radiation exposure, and the need for sustainable life support systems.",
    "How does the discovery of past water on Mars affect the search for extraterrestrial life?": "It increases the possibility that life may have existed on Mars when conditions were wetter and warmer.",
    "What role do Mars' moons play in its gravitational field?": "Phobos and Deimos slightly influence Mars' gravitational field and may provide information about the planet's history and the solar system's dynamics.",
    "How does the size of Mars' moons compare to other moons in the solar system?": "Phobos and Deimos are much smaller and less spherical than many other moons, such as Earth's Moon or Jupiter's Galilean moons.",
    "What scientific instruments are typically included on Mars rovers to study the planet?": "Instruments include cameras, spectrometers, drills, environmental sensors, and sometimes even laboratories for analyzing soil and rocks.",
    "How has the study of Mars contributed to our understanding of planetary atmospheres?": "Studying Mars' thin atmosphere provides insights into atmospheric loss processes and the factors that contribute to a planet's ability to retain an atmosphere.",
    "What evidence supports the idea that Mars had a thicker atmosphere in the past?": "Features like dried-up river valleys, lake beds, and mineral deposits that form in the presence of water support the idea of a thicker past atmosphere.",
    "How do the temperature variations on Mars affect potential future colonization efforts?": "Extreme and fluctuating temperatures would require habitats and life support systems capable of maintaining stable conditions for humans.",
    "What is the significance of Mars' rocky mantle in its overall structure?": "The rocky mantle contributes to Mars' geological activity and influences its magnetic field and surface geology.",
    "How does Mars' thin atmosphere impact its ability to support liquid water on the surface today?": "A thin atmosphere results in low pressure and temperature conditions that prevent liquid water from existing stably on the surface.",
    "What are the main differences between Mars' and Earth's atmospheres?": "Mars' atmosphere is much thinner, composed mostly of carbon dioxide, with trace amounts of nitrogen and argon, whereas Earth's atmosphere is thicker and rich in nitrogen and oxygen.",
    "How do the composition and structure of Mars' core compare to Earth's core?": "Both cores are primarily composed of iron and nickel, but Mars' core also contains sulfur and is smaller relative to the planet's size compared to Earth's core.",
    "What geological evidence suggests that Mars once had a denser atmosphere?": "The presence of ancient riverbeds, lake basins, and mineral deposits that form in wetter conditions suggest Mars once had a denser atmosphere.",
    "How does the size of Mars influence its gravitational pull compared to Earth?": "Mars' smaller size results in a weaker gravitational pull, about 38% of Earth's, affecting everything from atmosphere retention to human movement.",
    "What are the implications of Mars' gravitational strength for human missions?": "Lower gravity could impact human health over long periods and affect the design and operation of habitats and machinery.",
    "How does Mars' rotation period compare to Earth's day length?": "A day on Mars, called a sol, is approximately 24 hours and 39 minutes, slightly longer than an Earth day.",
    "What are the key elements found in Mars' crust and their significance?": "Elements like iron, magnesium, aluminum, calcium, and potassium are essential for understanding Mars' geological history and potential for past life.",
    "Why is studying Mars' thin atmosphere important for understanding climate change on other planets?": "It provides insights into how atmospheres can evolve and the factors that lead to significant climate changes over time.",
    "How do the atmospheric conditions on Mars affect the potential for human exploration?": "Thin atmosphere means less protection from radiation and extreme temperatures, requiring robust life support and habitat systems for human explorers.",
    "What technological challenges must be overcome to maintain human life on Mars?": "Challenges include creating sustainable life support systems, ensuring adequate shelter from radiation, providing reliable energy sources, and managing limited resources.",
    "How does the presence of sulfur in Mars' core influence its geological properties?": "Sulfur can affect the core's melting point and the planet's magnetic properties, influencing geological activity.",
    "What role do carbon dioxide, nitrogen, and argon play in Mars' atmosphere?": "Carbon dioxide is the dominant gas, contributing to greenhouse effects, while nitrogen and argon are trace gases that affect atmospheric pressure and composition.",
    "How does the red appearance of Mars compare to other celestial bodies in the solar system?": "Mars is uniquely red due to widespread iron oxide, whereas other celestial bodies have different colors based on their surface compositions and atmospheres.",
    "What future missions are planned to further explore Mars' environment and potential for life?": "Future missions include NASA's Perseverance rover, the European Space Agency's Rosalind Franklin rover, and potential crewed missions by NASA and SpaceX.",
    "How does the study of Mars' moons contribute to our understanding of its gravitational interactions?": "Analyzing Phobos and Deimos helps scientists understand Mars' gravitational pull and the dynamics of captured celestial bodies.",
    "What are the potential scientific benefits of sending more rovers to Mars?": "More rovers can explore diverse regions, collect more data on geology and climate, search for signs of past life, and test technologies for future human missions.",
    "How does the distance of Mars from the Sun influence its surface temperature and climate?": "Being 1.5 AU from the Sun results in lower solar energy reaching Mars, contributing to its colder surface temperatures and arid climate.",
    "What are the main challenges in interpreting data from Mars missions?": "Challenges include dealing with harsh environmental conditions, limited communication bandwidth, and the difficulty of inferring past conditions from current geological features.",
    "How does the presence of dust storms on Mars affect its atmosphere and surface?": "Dust storms can obscure visibility, alter atmospheric temperature, and impact the operation of rovers and other equipment on the surface.",
    "What evidence suggests that Mars may still have subsurface water ice?": "Observations of certain geological formations and the detection of hydrogen by orbiters suggest the presence of subsurface water ice on Mars.",
    "How does the axial tilt of Mars influence its seasons compared to Earth's?": "Mars has an axial tilt of about 25 degrees, similar to Earth's 23.4 degrees, resulting in comparable seasonal changes, though amplified by its thinner atmosphere.",
    "What are the implications of Mars' geological features for understanding planetary formation?": "Studying Mars' geology helps scientists learn about volcanic activity, tectonics, erosion, and impact history, providing insights into how rocky planets form and evolve.",
    "How do the environmental conditions on Mars challenge the design of scientific instruments?": "Instruments must withstand extreme temperatures, dust accumulation, radiation exposure, and mechanical stresses, requiring robust and resilient engineering solutions.",
    "What is the significance of Mars' rocky mantle in terms of geological activity?": "A rocky mantle suggests past volcanic activity and tectonic processes that have shaped the planet's surface and interior structure.",
    "How does the study of Mars contribute to our understanding of Earth's climate history?": "Comparing Mars' past warmer and wetter conditions with Earth's climate history can provide insights into atmospheric evolution and climate change mechanisms.",
    "What role do magnetic fields play in the atmospheric retention of planets, and how does this relate to Mars?": "Magnetic fields protect planets from solar wind stripping away their atmospheres. Mars lacks a strong global magnetic field, contributing to its thin atmosphere.",
    "How does the presence of argon in Mars' atmosphere compare to Earth's atmospheric composition?": "Mars' atmosphere contains argon as a trace gas, similar to Earth, but in much lower concentrations relative to its overall atmospheric composition.",
    "What are the potential resources on Mars that could support future human missions?": "Potential resources include water ice, carbon dioxide for fuel production, minerals for construction, and sunlight for energy generation.",
    "How do temperature gradients on Mars affect the potential for weather patterns?": "Significant temperature differences between day and night can drive wind and weather patterns, despite the thin atmosphere.",
    "What are the challenges of maintaining a stable temperature for equipment on Mars?": "Challenges include extreme cold, heat dissipation issues, and the need for insulation and active heating systems to protect sensitive equipment.",
    "How does the presence of nitrogen in Mars' atmosphere compare to Earth's atmosphere?": "While Earth's atmosphere contains about 78% nitrogen, Mars' atmosphere has only about 2.6% nitrogen, making it a minor component.",
    "What are the primary scientific objectives of current Mars rovers?": "Objectives include searching for signs of past life, analyzing soil and rock composition, studying climate and geology, and preparing for future human exploration.",
    "How does the thin atmosphere of Mars influence its weather systems?": "A thin atmosphere results in weaker weather systems, limited cloud formation, and less precipitation compared to Earth.",
    "What geological evidence supports the existence of ancient lakes on Mars?": "Features like dried riverbeds, lake basins, and sedimentary deposits observed by rovers and orbiters support the existence of ancient lakes on Mars.",
    "How does Mars' position in the solar system affect its orbital period around the Sun?": "Being the fourth planet from the Sun, Mars has an orbital period of about 687 Earth days, roughly 1.88 Earth years.",
    "What are the implications of Mars' orbital period for its seasons?": "Longer orbital periods result in longer seasons, with each season lasting about twice as long as those on Earth.",
    "How does the composition of Mars' atmosphere influence its potential for greenhouse warming?": "A thin atmosphere with a high concentration of carbon dioxide can still contribute to greenhouse warming, but less effectively than Earth's thicker atmosphere.",
    "What are the potential effects of Mars' dust storms on its climate and environment?": "Dust storms can raise surface temperatures by absorbing sunlight, obscure solar panels, and redistribute dust across the planet, affecting albedo and climate patterns.",
    "How does the size of Mars' core compare to that of other terrestrial planets?": "Mars' core is smaller relative to its overall size compared to Earth's larger and more massive core.",
    "What role does sulfur play in the composition and behavior of Mars' core?": "Sulfur lowers the melting point of the core materials, potentially allowing parts of Mars' core to remain molten and influence geological activity.",
    "How does the study of Mars' surface composition aid in understanding its geological history?": "Analyzing surface composition reveals information about volcanic activity, erosion processes, impact events, and the presence of past water, all contributing to understanding Mars' geological history.",
    "What technological advancements have been made to protect rovers from Mars' harsh conditions?": "Advancements include durable materials, thermal insulation, dust-resistant components, and autonomous navigation systems to ensure rover functionality and longevity.",
    "How does the low atmospheric pressure on Mars affect potential human habitats?": "Low pressure requires habitats to be pressurized and airtight to protect humans from the thin atmosphere and prevent air loss.",
    "What are the challenges of using solar power on Mars given its distance from the Sun?": "Reduced solar intensity means solar panels must be more efficient or larger to generate sufficient power, and dust accumulation can hinder energy production.",
    "How does the presence of potassium in Mars' crust influence its geological properties?": "Potassium can play a role in volcanic activity and the formation of minerals, affecting Mars' geological diversity.",
    "What evidence suggests that Mars once had a magnetic field?": "Remnants of magnetism in Martian rocks suggest that Mars once had a global magnetic field, which has since diminished.",
    "How does the study of Mars' moons contribute to understanding the planet's gravitational interactions?": "Examining the orbits and compositions of Phobos and Deimos helps scientists understand Mars' gravitational influence and the history of these moons.",
    "What are the primary components of Mars' thin atmosphere and their proportions?": "Mars' atmosphere is composed mostly of carbon dioxide (about 95.3%), with nitrogen (2.7%) and argon (1.6%) as minor components.",
    "How do the surface conditions on Mars compare to those of Earth's Moon?": "Mars has a thin atmosphere and more extreme temperature variations, while the Moon has no atmosphere and even more severe temperature fluctuations.",
    "What are the potential benefits of discovering subsurface water on Mars?": "Subsurface water could support future human missions, provide resources for life support, and offer habitats for microbial life.",
    "How does the composition of Mars' crust compare to Earth's crust?": "Mars' crust contains similar elements like iron, magnesium, aluminum, calcium, and potassium, but the proportions and mineral compositions may differ, reflecting different geological histories.",
    "What scientific instruments are essential for analyzing the composition of Mars' surface?": "Spectrometers, X-ray diffraction instruments, and cameras with mineralogical filters are essential for analyzing surface composition.",
    "How does Mars' gravity influence the behavior of its thin atmosphere?": "Lower gravity results in a thinner atmosphere, allowing gases to escape more easily and limiting the atmosphere's ability to retain heat and protect the surface.",
    "What are the implications of Mars' thin atmosphere for sound propagation?": "A thin atmosphere makes sound propagation on Mars very weak, meaning sounds would be faint and not travel far compared to Earth.",
    "How does the lack of a strong magnetic field affect Mars' atmosphere?": "Without a strong magnetic field, solar wind can strip away atmospheric particles, contributing to the thinning of Mars' atmosphere over time.",
    "What are the primary challenges in sustaining liquid water on Mars' surface today?": "Low atmospheric pressure and cold temperatures prevent liquid water from existing stably on the surface, leading to ice or vapor forms instead.",
    "How do the geological features on Mars provide clues about its past climate?": "Features like river valleys, lake beds, and mineral deposits indicate that Mars once had a warmer, wetter climate capable of supporting liquid water.",
    "What role does carbon dioxide play in Mars' current atmospheric conditions?": "Carbon dioxide is the dominant gas, contributing to the greenhouse effect and influencing the planet's temperature and weather patterns.",
    "How does the presence of nitrogen in Mars' atmosphere compare to its role on Earth?": "Nitrogen is a minor component on Mars (about 2.7%) and does not play as significant a role in atmospheric chemistry or pressure as it does on Earth.",
    "What are the primary sources of carbon dioxide in Mars' atmosphere?": "Carbon dioxide is primarily released from volcanic outgassing and sublimation of dry ice from the polar caps.",
    "How do the temperature extremes on Mars affect potential liquid water reservoirs?": "Extreme cold and low pressure cause any liquid water to quickly freeze or vaporize, making stable liquid reservoirs unlikely without subsurface protection.",
    "What are the potential geological processes that have shaped Mars' surface features?": "Volcanism, erosion by wind and water, impact cratering, and tectonic activity have all contributed to shaping Mars' surface.",
    "How does the thickness of Mars' atmosphere compare to that of Earth's?": "Mars' atmosphere is about 1% the thickness of Earth's atmosphere, resulting in much lower pressure and density.",
    "What are the main challenges in detecting signs of past life on Mars?": "Challenges include preserving and accurately identifying biosignatures, accessing suitable geological formations, and differentiating between biological and abiotic processes.",
    "How does the study of Mars inform our understanding of planetary habitability?": "Studying Mars helps identify the conditions necessary for life, the processes that affect habitability, and the potential for life beyond Earth.",
    "What are the key differences between Phobos and Deimos?": "Phobos is larger, closer to Mars, and orbits faster than Deimos, which is smaller and has a more distant orbit.",
    "How does the gravitational interaction between Mars and its moons affect their orbits?": "Gravitational interactions can cause tidal forces, orbital decay (especially for Phobos), and influence the moons' rotational dynamics.",
    "What future technologies are being developed to support human missions to Mars?": "Technologies include advanced life support systems, habitat modules, in-situ resource utilization (ISRU) technologies, and reliable propulsion systems.",
    "How does Mars' position in the solar system influence its orbital speed and period?": "Being farther from the Sun, Mars has a slower orbital speed and a longer orbital period compared to inner planets like Earth.",
    "What are the scientific goals of the upcoming Mars Sample Return missions?": "Goals include analyzing Martian soil and rock samples for signs of past life, understanding geological history, and studying the planet's climate evolution.",
    "How does the lack of a thick atmosphere on Mars impact its surface erosion processes?": "With a thin atmosphere, erosion is primarily driven by wind and dust storms, which are less intense than Earth's water-driven erosion but can still significantly shape the surface over time.",
    "What are the implications of Mars' geological diversity for its history?": "Geological diversity indicates a complex history involving volcanic activity, tectonics, erosion, and impacts, reflecting the planet's dynamic past.",
    "How does the study of Mars' surface minerals aid in understanding its environmental history?": "Analyzing surface minerals reveals information about past water activity, volcanic processes, and atmospheric conditions, helping reconstruct Mars' environmental history.",
    "What are the primary factors that have led to the current state of Mars' atmosphere?": "Factors include atmospheric escape due to solar wind, lack of a strong magnetic field, volcanic outgassing, and the sequestration of gases into the surface and interior.",
    "How does the presence of argon in Mars' atmosphere compare to its abundance on Earth?": "Argon is present in both atmospheres, but it is a trace gas on Mars, making up about 1.6% of its atmosphere compared to about 0.93% on Earth.",
    "What are the potential benefits of discovering active geological processes on Mars?": "Active geological processes could indicate ongoing heat flow, potential habitats for life, and dynamic changes that affect surface conditions.",
    "How does Mars' axial tilt contribute to its seasonal weather patterns?": "Mars' axial tilt of about 25 degrees causes it to experience seasons similar to Earth, with variations in temperature and weather patterns throughout its orbital cycle.",
    "What are the primary components of Mars' thin atmosphere and their roles?": "Carbon dioxide contributes to greenhouse warming, nitrogen provides minor atmospheric pressure, and argon plays a role in thermal properties and gas dynamics.",
    "How does the thin atmosphere of Mars affect sound transmission on its surface?": "A thin atmosphere results in very weak sound transmission, making sounds on Mars faint and unable to travel far distances.",
    "What are the main challenges in designing habitats for humans on Mars?": "Challenges include providing adequate protection from radiation, maintaining atmospheric pressure and temperature, ensuring reliable life support systems, and utilizing local resources.",
    "How does the distance of Mars from the Sun affect mission planning and communication?": "Greater distance requires longer travel times, increased communication delays, and more robust systems to handle the harsher environment and longer mission durations.",
    "What are the key differences between Phobos and Deimos in terms of their orbits?": "Phobos orbits closer to Mars and has a faster orbital period, completing an orbit approximately every 7 hours, while Deimos orbits farther out with a period of about 30.3 hours.",
    "How does the study of Mars' geological features contribute to our understanding of planetary evolution?": "It provides insights into processes like volcanism, erosion, impact cratering, and tectonics, which are fundamental to understanding how planets develop and change over time.",
    "What role does magnesium play in the composition of Mars' crust?": "Magnesium is a key component of silicate minerals, contributing to the structural integrity and geological activity of Mars' crust.",
    "How does the presence of potassium in Mars' crust affect its potential for supporting life?": "Potassium is an essential nutrient for life as we know it, and its presence in Mars' crust could be indicative of past biological or geological processes favorable to life.",
    "What are the potential sources of argon in Mars' atmosphere?": "Argon in Mars' atmosphere likely originates from volcanic outgassing and the radioactive decay of potassium-40 in the crust.",
    "How does the distribution of iron in Mars' surface materials influence its magnetic properties?": "Iron-rich minerals can contribute to localized magnetic fields, but without a global magnetic field, Mars' overall magnetic properties are weak and fragmented.",
    "What are the implications of Mars' thin atmosphere for solar-powered missions?": "Solar-powered missions must account for reduced solar energy availability and potential dust accumulation on solar panels, requiring efficient energy management and cleaning mechanisms.",
    "How does the presence of sulfur in Mars' core influence volcanic activity?": "Sulfur lowers the melting point of core materials, potentially enhancing volcanic activity by allowing magma to form more easily.",
    "What are the challenges of detecting water ice in Mars' polar regions?": "Challenges include the extreme cold, limited accessibility, and the need for sensitive instruments to differentiate between water ice and other materials.",
    "How does Mars' thin atmosphere affect its surface weather compared to Earth?": "Mars has weaker and less frequent weather events, primarily driven by wind and dust storms, without the influence of liquid water-based weather systems like on Earth.",
    "What are the primary differences between the cores of Earth and Mars?": "Earth's core is larger relative to its size, has a liquid outer core generating a magnetic field, and contains more sulfur, while Mars' core is smaller and may be partly molten without a significant global magnetic field.",
    "How does the presence of iron oxide on Mars' surface relate to its geological history?": "Iron oxide indicates past oxidation processes, likely involving the presence of liquid water and a thicker atmosphere, which facilitated the rusting of iron-rich minerals.",
    "What are the potential scientific benefits of studying Mars' crust composition?": "Studying the crust composition helps understand the planet's geological history, volcanic activity, mineral diversity, and potential resources for future missions.",
    "How does the low gravity on Mars influence the behavior of its atmosphere and weather systems?": "Low gravity results in a thinner atmosphere, weaker weather systems, and less ability to retain heat, contributing to extreme temperature variations and limited atmospheric circulation.",
    "What technological solutions are being developed to mitigate the effects of Mars' harsh environment on equipment?": "Solutions include thermal insulation, radiation shielding, dust-resistant coatings, autonomous cleaning systems, and robust mechanical designs to ensure equipment longevity and functionality.",
    "How does the presence of nitrogen in Mars' atmosphere influence its chemical processes?": "Nitrogen acts as a minor inert gas, playing a limited role in chemical reactions but contributing to the overall atmospheric composition and pressure.",
    "What are the key factors that have led to the current thinness of Mars' atmosphere?": "Factors include atmospheric escape due to solar wind, lack of a strong global magnetic field, and sequestration of gases into the surface and interior over time.",
    "How does the study of Mars' atmosphere contribute to our understanding of climate dynamics on other planets?": "It provides a comparative perspective on how different atmospheric compositions and planetary characteristics influence climate, weather patterns, and atmospheric evolution.",
    "What are the potential methods for utilizing Mars' atmospheric carbon dioxide for future missions?": "Methods include converting CO2 into oxygen for life support, producing rocket fuel through the Sabatier process, and using it in greenhouse agriculture systems.",
    "How do the size and composition of Phobos and Deimos compare to typical asteroids in the solar system?": "Phobos and Deimos are smaller and less dense than many typical asteroids, with compositions that suggest a mix of rocky and carbonaceous materials similar to certain asteroid types.",
    "What scientific questions remain unanswered about Mars' geological and atmospheric history?": "Questions include the exact timeline of atmospheric loss, the extent and duration of past water activity, the presence of a past global magnetic field, and the potential for past or present life.",
    "How does the study of Mars inform the search for habitable exoplanets?": "Insights from Mars' habitability challenges and successes help refine criteria for habitable conditions on exoplanets, guiding the search for similar environments in other star systems.",
    "What are the key differences between the geological features of Mars and those of Earth?": "Mars lacks active plate tectonics, has large shield volcanoes like Olympus Mons, extensive canyon systems like Valles Marineris, and a thinner, oxidized crust compared to Earth's diverse geological features driven by active plate tectonics and a robust magnetic field.",
    "How does the presence of argon in Mars' atmosphere affect its thermal properties?": "Argon, being a noble gas, has minimal direct impact on thermal properties but contributes to the overall composition"
}

# Precompute the embeddings for all contexts
context_descriptions = list(contexts.keys())
context_texts = list(contexts.values())
context_embeddings = embedding_model.encode(context_descriptions, convert_to_tensor=True)

# Chatbot loop
def chatbot():
    print("Hello! Ask me anything about Mercury.")
    
    while True:
        question = input("You: ")

        # Exit if the user types "exit" or "quit"
        if question.lower() in ['exit', 'quit']:
            print("Goodbye!")
            break

        # Encode the user's question
        question_embedding = embedding_model.encode(question, convert_to_tensor=True)

        # Compute cosine similarity between the question and context descriptions
        similarities = util.pytorch_cos_sim(question_embedding, context_embeddings)

        # Find the index of the most similar context
        best_match_idx = torch.argmax(similarities)

        # Retrieve the most relevant context
        best_context_description = context_descriptions[best_match_idx]
        best_context_text = contexts[best_context_description]

        # Get the answer from the model using the selected context
        qa_input = {
            "question": question,
            "context": best_context_text
        }
        result = nlp(qa_input, max_answer_len=100)

        # Print the answer
        print(f"Chatbot: {result['answer']}")

# Run the chatbot
if __name__ == "__main__":
    chatbot()
