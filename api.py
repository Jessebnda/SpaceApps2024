# app.py
from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import pipeline
from sentence_transformers import SentenceTransformer, util
import torch

app = Flask(__name__)
CORS(app)

# Cargar modelos
model_name = "distilbert-base-cased-distilled-squad"
nlp = pipeline('question-answering', model=model_name, tokenizer=model_name)
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Contextos predefinidos con descripciones correspondientes
contexts = {
    """
    Mercury is the smallest planet in our solar system and nearest to the Sun, Mercury is only slightly larger than Earth's Moon. From the surface of Mercury, the Sun would appear more than three times as large as it does when viewed from Earth, and the sunlight would be as much as seven times brighter.
Mercury's surface temperatures are both extremely hot and cold. Because the planet is so close to the Sun, day temperatures can reach highs of 800°F (430°C). Without an atmosphere to retain that heat at night, temperatures can dip as low as -290°F (-180°C).
Despite its proximity to the Sun, Mercury is not the hottest planet in our solar system  that title belongs to nearby Venus, thanks to its dense atmosphere. But Mercury is the fastest planet, zipping around the Sun every 88 Earth days.
With a radius of 1,516 miles (2,440 kilometers), Mercury is a little more than 1/3 the width of Earth. If Earth were the size of a nickel, Mercury would be about as big as a blueberry.
From an average distance of 36 million miles (58 million kilometers), Mercury is 0.4 astronomical units away from the Sun. One astronomical unit (abbreviated as AU), is the distance from the Sun to Earth. From this distance, it takes sunlight 3.2 minutes to travel from the Sun to Mercury.
Mercury spins slowly on its axis and completes one rotation every 59 Earth days. But when Mercury is moving fastest in its elliptical orbit around the Sun (and it is closest to the Sun), each rotation is not accompanied by sunrise and sunset like it is on most other planets. The morning Sun appears to rise briefly, set, and rise again from some parts of the planet's surface. The same thing happens in reverse at sunset for other parts of the surface. One Mercury solar day (one full day-night cycle) equals 176 Earth days just over two years on Mercury.
Mercury is the second densest planet, after Earth. It has a large metallic core with a radius of about 1,289 miles (2,074 kilometers), about 85% of the planet's radius. There is evidence that it is partly molten or liquid. Mercury's outer shell, comparable to Earth's outer shell (called the mantle and crust), is only about 400 kilometers (250 miles) thick.
Mercury's surface resembles that of Earth's Moon, scarred by many impact craters resulting from collisions with meteoroids and comets. Craters and features on Mercury are named after famous deceased artists, musicians, or authors, including children's author Dr. Seuss and dance pioneer Alvin Ailey.
Very large impact basins, including Caloris (960 miles or 1,550 kilometers in diameter) and Rachmaninoff (190 miles, or 306 kilometers in diameter), were created by asteroid impacts on the planet's surface early in the solar system's history. While there are large areas of smooth terrain, there are also cliffs, some hundreds of miles long and soaring up to a mile high. They rose as the planet's interior cooled and contracted over the billions of years since Mercury formed.
Most of Mercury's surface would appear greyish-brown to the human eye. The bright streaks are called "crater rays." They are formed when an asteroid or comet strikes the surface. The tremendous amount of energy that is released in such an impact digs a big hole in the ground, and also crushes a huge amount of rock under the point of impact. Some of this crushed material is thrown far from the crater and then falls to the surface, forming the rays. Fine particles of crushed rock are more reflective than large pieces, so the rays look brighter. The space environment dust impacts and solar-wind particles causes the rays to darken with time.
Temperatures on Mercury are extreme. During the day, temperatures on the surface can reach 800 degrees Fahrenheit (430 degrees Celsius). Because the planet has no atmosphere to retain that heat, nighttime temperatures on the surface can drop to minus 290 degrees Fahrenheit (minus 180 degrees Celsius).
Mercury may have water ice at its north and south poles inside deep craters, but only in regions in permanent shadows. In those shadows, it could be cold enough to preserve water ice despite the high temperatures on sunlit parts of the planet.

Venus is the second planet from the Sun, and Earth's closest planetary neighbor. Venus is the third brightest object in the sky after the Sun and Moon. Venus spins slowly in the opposite direction from most planets.
Venus is similar in structure and size to Earth, and is sometimes called Earth's evil twin. Its thick atmosphere traps heat in a runaway greenhouse effect, making it the hottest planet in our solar system with surface temperatures hot enough to melt lead. Below the dense, persistent clouds, the surface has volcanoes and deformed mountains.
Venus orbits the Sun from an average distance of 67 million miles (108 million kilometers), or 0.72 astronomical units. One astronomical unit (abbreviated as AU), is the distance from the Sun to Earth. From this distance, it takes sunlight about six minutes to travel from the Sun to Venus.
Earth's nearness to Venus is a matter of perspective. The planet is nearly as big around as Earth. Its diameter at its equator is about 7,521 miles (12,104 kilometers), versus 7,926 miles (12,756 kilometers) for Earth. From Earth, Venus is the brightest object in the night sky after our own Moon. The ancients, therefore, gave it great importance in their cultures, even thinking it was two objects: a morning star and an evening star. That’s where the trick of perspective comes in.
Spending a day on Venus would be quite a disorienting experience - that is, if your spacecraft or spacesuit could protect you from temperatures in the range of 900 degrees Fahrenheit (475 Celsius). For one thing, your “day” would be 243 Earth days long – longer even than a Venus year (one trip around the Sun), which takes only 225 Earth days. For another, because of the planet's extremely slow rotation, sunrise to sunset would take 117 Earth days. And by the way, the Sun would rise in the west and set in the east, because Venus spins backward compared to Earth.
While you’re waiting, don’t expect any seasonal relief from the unrelenting temperatures. On Earth, with its spin axis tilted by about 23 degrees, we experience summer when our part of the planet (our hemisphere) receives the Sun’s rays more directly – a result of that tilt. In winter, the tilt means the rays are less direct. No such luck on Venus: Its very slight tilt is only three degrees, which is too little to produce noticeable seasons.
If we could slice Venus and Earth in half, pole to pole, and place them side by side, they would look remarkably similar. Each planet has an iron core enveloped by a hot-rock mantle; the thinnest of skins forms a rocky, exterior crust. On both planets, this thin skin changes form and sometimes erupts into volcanoes in response to the ebb and flow of heat and pressure deep beneath.
On Earth, the slow movement of continents over thousands and millions of years reshapes the surface, a process known as “plate tectonics.” Something similar might have happened on Venus early in its history. Today a key element of this process could be operating: subduction, or the sliding of one continental “plate” beneath another, which can also trigger volcanoes. Subduction is believed to be the first step in creating plate tectonics.


While Earth is only the fifth largest planet in the solar system, it is the only world in our solar system with liquid water on the surface. Just slightly larger than nearby Venus, Earth is the biggest of the four planets closest to the Sun, all of which are made of rock and metal.
With an equatorial diameter of 7926 miles (12,760 kilometers), 
Earth is the biggest of the terrestrial planets and the fifth largest planet in our solar system.
From an average distance of 93 million miles (150 million kilometers), Earth is exactly one astronomical unit away from the Sun because one astronomical unit (abbreviated as AU), is the distance from the Sun to Earth. This unit provides an easy way to quickly compare planets' distances from the Sun.
It takes about eight minutes for light from the Sun to reach our planet.
As Earth orbits the Sun, it completes one rotation every 23.9 hours. It takes 365.25 days to complete one trip around the Sun. That extra quarter of a day presents a challenge to our calendar system, which counts one year as 365 days. To keep our yearly calendars consistent with our orbit around the Sun, every four years we add one day. That day is called a leap day, and the year it's added to is called a leap year.
Earth's axis of rotation is tilted 23.4 degrees with respect to the plane of Earth's orbit around the Sun. This tilt causes our yearly cycle of seasons. During part of the year, the northern hemisphere is tilted toward the Sun, and the southern hemisphere is tilted away. With the Sun higher in the sky, solar heating is greater in the north producing summer there. Less direct solar heating produces winter in the south. Six months later, the situation is reversed. When spring and fall begin, both hemispheres receive roughly equal amounts of heat from the Sun.
Earth is composed of four main layers, starting with an inner core at the planet's center, enveloped by the outer core, mantle, and crust.
The inner core is a solid sphere made of iron and nickel metals about 759 miles (1,221 kilometers) in radius. There the temperature is as high as 9,800 degrees Fahrenheit (5,400 degrees Celsius). Surrounding the inner core is the outer core. This layer is about 1,400 miles (2,300 kilometers) thick, made of iron and nickel fluids.
In between the outer core and crust is the mantle, the thickest layer. This hot, viscous mixture of molten rock is about 1,800 miles (2,900 kilometers) thick and has the consistency of caramel. The outermost layer, Earth's crust, goes about 19 miles (30 kilometers) deep on average on land. At the bottom of the ocean, the crust is thinner and extends about 3 miles (5 kilometers) from the seafloor to the top of the mantle.
Near the surface, Earth has an atmosphere that consists of 78% nitrogen, 21% oxygen, and 1% other gases such as argon, carbon dioxide, and neon. The atmosphere affects Earth's long-term climate and short-term local weather and shields us from much of the harmful radiation coming from the Sun. It also protects us from meteoroids, most of which burn up in the atmosphere, seen as meteors in the night sky, before they can strike the surface as meteorites.

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

Jupiter's signature stripes and swirls are actually cold, windy clouds of ammonia and water, floating in an atmosphere of hydrogen and helium. The dark orange stripes are called belts, while the lighter bands are called zones, and they flow east and west in opposite directions. Jupiter’s iconic Great Red Spot is a giant storm bigger than Earth that has raged for hundreds of years.The king of planets was named for Jupiter, king of the gods in Roman mythology. Most of its moons are also named for mythological characters, figures associated with Jupiter or his Greek counterpart, Zeus.
With a radius of 43,440.7 miles (69,911 kilometers), Jupiter is 11 times wider than Earth. If Earth were the size of a grape, Jupiter would be about as big as a basketball.
From an average distance of 484 million miles (778 million kilometers), Jupiter is 5.2 astronomical units away from the Sun. One astronomical unit (abbreviated as AU), is the distance from the Sun to Earth. From this distance, it takes sunlight 43 minutes to travel from the Sun to Jupiter.
Jupiter has the shortest day in the solar system. One day on Jupiter takes only about 10 hours (the time it takes for Jupiter to rotate or spin around once), and Jupiter makes a complete orbit around the Sun (a year in Jovian time) in about 12 Earth years (4,333 Earth days).
Jupiter has 95 moons that are officially recognized by the International Astronomical Union. The four largest moons – Io, Europa, Ganymede, and Callisto – were first observed by the astronomer Galileo Galilei in 1610 using an early version of the telescope. These four moons are known today as the Galilean satellites, and they're some of the most fascinating destinations in our solar system.
The composition of Jupiter is similar to that of the Sun – mostly hydrogen and helium. Deep in the atmosphere, pressure and temperature increase, compressing the hydrogen gas into a liquid. This gives Jupiter the largest ocean in the solar system – an ocean made of hydrogen instead of water. Scientists think that, at depths perhaps halfway to the planet's center, the pressure becomes so great that electrons are squeezed off the hydrogen atoms, making the liquid electrically conducting like metal. Jupiter's fast rotation is thought to drive electrical currents in this region, with the spinning of the liquid metallic hydrogen acting like a dynamo, generating the planet's powerful magnetic field.
The vivid colors you see in thick bands across Jupiter may be plumes of sulfur and phosphorus-containing gases rising from the planet's warmer interior. Jupiter's fast rotation – spinning once every 10 hours – creates strong jet streams, separating its clouds into dark belts and bright zones across long stretches.
With no solid surface to slow them down, Jupiter's spots can persist for many years. Stormy Jupiter is swept by over a dozen prevailing winds, some reaching up to 335 miles per hour (539 kilometers per hour) at the equator. The Great Red Spot, a swirling oval of clouds twice as wide as Earth, has been observed on the giant planet for more than 300 years. More recently, three smaller ovals merged to form the Little Red Spot, about half the size of its larger cousin.

Saturn is the sixth planet from the Sun, and the second-largest planet in our solar system.
Like fellow gas giant Jupiter, Saturn is a massive ball made mostly of hydrogen and helium. Saturn is not the only planet to have rings, but none are as spectacular or as complex as Saturn's. Saturn also has dozens of moons.
From the jets of water that spray from Saturn's moon Enceladus to the methane lakes on smoggy Titan, the Saturn system is a rich source of scientific discovery and still holds many mysteries.
With an equatorial diameter of about 74,897 miles (120,500 kilometers), Saturn is 9 times wider than Earth. If Earth were the size of a nickel, Saturn would be about as big as a volleyball.
From an average distance of 886 million miles (1.4 billion kilometers), Saturn is 9.5 astronomical units away from the Sun. One astronomical unit (abbreviated as AU), is the distance from the Sun to Earth. From this distance, it takes sunlight 80 minutes to travel from the Sun to Saturn.
Saturn has the second-shortest day in the solar system. One day on Saturn takes only 10.7 hours (the time it takes for Saturn to rotate or spin around once), and Saturn makes a complete orbit around the Sun (a year in Saturnian time) in about 29.4 Earth years (10,756 Earth days).
Saturn is home to a vast array of intriguing and unique worlds. From the haze-shrouded surface of Titan to crater-riddled Phoebe, each of Saturn's moons tells another piece of the story surrounding the Saturn system. As of June 8, 2023, Saturn has 146 moons in its orbit, with others continually awaiting confirmation of their discovery and official naming by the International Astronomical Union (IAU).
Saturn's rings are thought to be pieces of comets, asteroids, or shattered moons that broke up before they reached the planet, torn apart by Saturn's powerful gravity. They are made of billions of small chunks of ice and rock coated with other materials such as dust. The ring particles mostly range from tiny, dust-sized icy grains to chunks as big as a house. A few particles are as large as mountains. The rings would look mostly white if you looked at them from the cloud tops of Saturn, and interestingly, each ring orbits at a different speed around the planet.
Saturn is blanketed with clouds that appear as faint stripes, jet streams, and storms. The planet is many different shades of yellow, brown, and gray.
Winds in the upper atmosphere reach 1,600 feet per second (500 meters per second) in the equatorial region. In contrast, the strongest hurricane-force winds on Earth top out at about 360 feet per second (110 meters per second). And the pressure – the same kind you feel when you dive deep underwater – is so powerful it squeezes gas into a liquid.

Uranus is the seventh planet from the Sun, and it has the third largest diameter of planets in our solar system. Uranus appears to spin sideways.
Uranus is a very cold and windy world. The ice giant is surrounded by 13 faint rings and 28 small moons. Uranus rotates at a nearly 90-degree angle from the plane of its orbit. This unique tilt makes Uranus appear to spin sideways, orbiting the Sun like a rolling ball.
Uranus was the first planet found with the aid of a telescope. It was discovered in 1781 by astronomer William Herschel, although he originally thought it was either a comet or a star. It was two years later that the object was universally accepted as a new planet, in part because of observations by astronomer Johann Elert Bode.
With an equatorial diameter of 31,763 miles (51,118 kilometers), Uranus is four times wider than Earth. If Earth was the size of a nickel, Uranus would be about as big as a softball.
From an average distance of 1.8 billion miles (2.9 billion kilometers), Uranus is about 19 astronomical units away from the Sun. One astronomical unit (abbreviated as AU), is the distance from the Sun to Earth. From this distance, it takes sunlight 2 hours and 40 minutes to travel from the Sun to Uranus.
One day on Uranus takes about 17 hours. This is the amount of time it takes Uranus to rotate, or spin once around its axis. Uranus makes a complete orbit around the Sun (a year in Uranian time) in about 84 Earth years (30,687 Earth days).
Uranus is the only planet whose equator is nearly at a right angle to its orbit, with a tilt of 97.77 degrees. This may be the result of a collision with an Earth-sized object long ago. This unique tilt causes Uranus to have the most extreme seasons in the solar system. For nearly a quarter of each Uranian year, the Sun shines directly over each pole, plunging the other half of the planet into a 21-year-long, dark winter.
Uranus has two sets of rings. The inner system of nine rings consists mostly of narrow, dark grey rings. There are two outer rings: the innermost one is reddish like dusty rings elsewhere in the solar system, and the outer ring is blue like Saturn's E ring.
Uranus is one of two ice giants in the outer solar system (the other is Neptune). Most (80% or more) of the planet's mass is made up of a hot dense fluid of "icy" materials – water, methane, and ammonia – above a small rocky core. Near the core, it heats up to 9,000 degrees Fahrenheit (4,982 degrees Celsius).
As an ice giant, Uranus doesn’t have a true surface. The planet is mostly swirling fluids. While a spacecraft would have nowhere to land on Uranus, it wouldn’t be able to fly through its atmosphere unscathed either. The extreme pressures and temperatures would destroy a metal spacecraft.

Neptune is the eighth and most distant planet in our solar system.
Dark, cold, and whipped by supersonic winds, ice giant Neptune is more than 30 times as far from the Sun as Earth. Neptune is the only planet in our solar system not visible to the naked eye. In 2011 Neptune completed its first 165-year orbit since its discovery in 1846.
Neptune is so far from the Sun that high noon on the big blue planet would seem like dim twilight to us. The warm light we see here on our home planet is roughly 900 times as bright as sunlight on Neptune.
With an equatorial diameter of 30,775 miles (49,528 kilometers), Neptune is about four times wider than Earth. If Earth were the size of a nickel, Neptune would be about as big as a baseball.
From an average distance of 2.8 billion miles (4.5 billion kilometers), Neptune is 30 astronomical units away from the Sun. One astronomical unit (abbreviated as AU), is the distance from the Sun to Earth. From this distance, it takes sunlight 4 hours to travel from the Sun to Neptune.
One day on Neptune takes about 16 hours (the time it takes for Neptune to rotate or spin once). And Neptune makes a complete orbit around the Sun (a year in Neptunian time) in about 165 Earth years (60,190 Earth days).
Neptune has 16 known moons. Neptune's largest moon Triton was discovered on Oct. 10, 1846, by William Lassell, just 17 days after Johann Gottfried Galle discovered the planet. Since Neptune was named for the Roman god of the sea, its moons are named for various lesser sea gods and nymphs in Greek mythology.
Neptune has at least five main rings and four prominent ring arcs that we know of so far. Starting near the planet and moving outward, the main rings are named Galle, Leverrier, Lassell, Arago, and Adams. The rings are thought to be relatively young and short-lived.
Neptune is one of two ice giants in the outer solar system (the other is Uranus). Most (80% or more) of the planet's mass is made up of a hot dense fluid of "icy" materials – water, methane, and ammonia – above a small, rocky core. Of the giant planets, Neptune is the densest.
Neptune does not have a solid surface. Its atmosphere (made up mostly of hydrogen, helium, and methane) extends to great depths, gradually merging into water and other melted ices over a heavier, solid core with about the same mass as Earth.

Pluto and other dwarf planets are a lot like regular planets. So what’s the big difference? The International Astronomical Union (IAU), a world organization of astronomers, came up with the definition of a planet in 2006. According to the IAU, a planet must do three things:
Orbit its host star (In our solar system that’s the Sun).
Be mostly round. Be big enough that its gravity cleared away any other objects of similar size near its orbit around the Sun.
Pluto is by far the most famous dwarf planet. Discovered by Clyde Tombaugh in 1930, Pluto was long considered our solar system's ninth planet. But after other astronomers found similar intriguing worlds deeper in the distant Kuiper Belt – the IAU reclassified Pluto as a dwarf planet in 2006. 
There was widespread outrage on behalf of the demoted planet. Textbooks were updated, and the internet spawned memes with Pluto going through a range of emotions, from anger to loneliness.
On July 14, 2015, NASA’s New Horizons spacecraft made its historic flight through the Pluto system – providing the first close-up images of Pluto and its moons and collecting other data that has transformed our understanding of these mysterious worlds on the solar system’s outer frontier.
Dwarf planet Ceres is closer to home. Ceres is the largest object in the asteroid belt between Mars and Jupiter, and it's the only dwarf planet located in the inner solar system. Like Pluto, Ceres also was once classified as a planet. Ceres was the first dwarf planet to be visited by a spacecraft – NASA’s Dawn mission. 


Our Moon shares a name with all moons simply because people didn't know other moons existed until Galileo Galilei discovered four moons orbiting Jupiter in 1610. In Latin, the Moon was called Luna, which is the main adjective for all things Moon-related: lunar.
With a radius of about 1,080 miles (1,740 kilometers), the Moon is less than a third of the width of Earth. If Earth were the size of a nickel, the Moon would be about as big as a coffee bean.
The Moon is an average of 238,855 miles (384,400 kilometers) away. That means 30 Earth-sized planets could fit in between Earth and the Moon.
The Moon is slowly moving away from Earth, getting about an inch farther away each year.
The Moon is rotating at the same rate that it revolves around Earth (called synchronous rotation), so the same hemisphere faces Earth all the time. Some people call the far side – the hemisphere we never see from Earth – the "dark side", but that's misleading.
As the Moon orbits Earth, different parts are in sunlight or darkness at different times. The changing illumination is why, from our perspective, the Moon goes through phases. During a "full moon," the hemisphere of the Moon we can see from Earth is fully illuminated by the Sun. And a "new moon" occurs when the far side of the Moon has full sunlight, and the side facing us is having its night.
The Moon makes a complete orbit around Earth in 27 Earth days and rotates or spins at that same rate, or in that same amount of time. Because Earth is moving as well – rotating on its axis as it orbits the Sun – from our perspective, the Moon appears to orbit us every 29 days.
The Moon likely formed after a Mars-sized body collided with Earth several billion years ago.
The resulting debris from both Earth and the impactor accumulated to form our natural satellite 239,000 miles (384,000 kilometers) away. The newly formed Moon was in a molten state, but within about 100 million years, most of the global "magma ocean" had crystallized, with less-dense rocks floating upward and eventually forming the lunar crust.
Over billions of years, these impacts have ground up the surface of the Moon into fragments ranging from huge boulders to powder. Nearly the entire Moon is covered by a rubble pile of charcoal-gray, powdery dust, and rocky debris called the lunar regolith. Beneath is a region of fractured bedrock referred to as the megaregolith.
In October 2020, NASA’s Stratospheric Observatory for Infrared Astronomy (SOFIA) confirmed, for the first time, water on the sunlit surface of the Moon. This discovery indicates that water may be distributed across the lunar surface, and not limited to cold, shadowed places. SOFIA detected water molecules (H2O) in Clavius Crater, one of the largest craters visible from Earth, located in the Moon’s southern hemisphere.

Halley's Comet is a periodic comet, meaning it orbits the sun and returns to the inner solar system on a regular basis. It takes 75–76 years to orbit the sun and return to Earth, where it can be seen by the naked eye. 
Halley's Comet is about 11 kilometers in diameter, making it larger than 99% of asteroids and comparable in size to Boston. 
Halley's Comet is made up of water and gases like methane, ammonia, and carbon dioxide. 
As Halley's Comet approaches the sun, the ice and gases inside it expand and form a tail that can look like a shooting star. Comets have two tails, a dust tail and an ion (gas) tail. 
Halley's Comet was the first comet whose return was predicted, proving that some comets are part of our solar system. Edmond Halley predicted its return in 1758 after showing that comets seen in 1531, 1607, and 1682 were actually the same comet. 
Halley's Comet's next appearance is predicted to be in July 2061

Apophis is about 375 meters wide, roughly the size of a cruise liner. 
Apophis is an Aten asteroid, which means it has an orbit with a semi-major axis of less than one astronomical unit. 
Apophis's orbit takes it around the sun every 323,513 days, and it crosses Earth's orbit twice per revolution. 
On April 13, 2029, Apophis will pass within 32,000 kilometers of Earth's surface. It will be visible to the naked eye for about two billion people in parts of Asia, Europe, and Africa. 
In 2036, Apophis will pass more than 49 million kilometers from Earth. In 2116, Apophis could pass as close as 150,000 kilometers from Earth. 
NASA says that Apophis poses no threat to Earth for at least the next century, however they are constantly keeping an eye on it, in case its orbit changes.
Apophis was discovered in 2004. 
Apophis is also known as Apep, the great serpent and enemy of the Egyptian sun god Ra. 

The asteroid belt is a region within the solar system occupied by asteroids that are sparsely held together by gravity and occupying a region taking the shape of a gradient ring orbiting the Sun. Asteroids are small rocky bodies sometimes composed of iron and nickel, which orbit the Sun. The asteroid belt exists between the orbits of Mars and Jupiter, between 330 million and 480 million kilometers from the Sun.
Location: The asteroid belt is located between the orbits of Mars and Jupiter 
Size: The asteroid belt is a torus-shaped or disc-shaped region that contains millions or billions of asteroids 
Composition: The asteroids in the belt are made up of carbon-rich materials and ices 
Formation: The asteroid belt formed about 4.5 billion years ago, along with the rest of the solar system 
Asteroids: Over 600,000 asteroids in the belt have been identified and named 
Ceres: Ceres is the largest known asteroid at 620 miles across 
Density: The asteroid belt is not as densely packed as fiction often portrays it 
Color: The asteroid belt is largely empty space, so there isn't really much color to see at all 
Resonances: The mean distances of the asteroids are not uniformly distributed but exhibit population depletions, or “gaps”

"What is the smallest planet in our solar system?": "Mercury is the smallest planet in our solar system.",
    "How does the size of Mercury compare to Earth's Moon?": "Mercury is only slightly larger than Earth's Moon.",
    "How much brighter is sunlight on Mercury compared to Earth?": "Sunlight on Mercury is up to seven times brighter than on Earth.",
    "What are the temperature extremes on Mercury?": "Daytime temperatures on Mercury can reach 800°F (430°C), while nighttime temperatures can drop to -290°F (-180°C).",
    "Why isn’t Mercury the hottest planet despite being closest to the Sun?": "Venus is the hottest planet due to its dense atmosphere, which traps heat more effectively than Mercury's lack of atmosphere.",
    "How long does it take Mercury to orbit the Sun?": "Mercury orbits the Sun every 88 Earth days.",
    "How long is a solar day on Mercury?": "A solar day on Mercury lasts 176 Earth days.",
    "What is the composition of Mercury's core?": "Mercury has a large metallic core, making up about 85% of its radius, and it is partly molten or liquid.",
    "What are the 'crater rays' on Mercury's surface?": "Crater rays are bright streaks formed when an asteroid or comet strikes Mercury’s surface, scattering reflective crushed material.",
    "Can Mercury have water ice?": "Yes, water ice may exist in deep craters at Mercury’s poles in regions that are permanently shadowed, despite the extreme temperatures on other parts of the planet.",
    "Which planet is the second from the Sun?": "Venus is the second planet from the Sun.",
    "What is Earth's closest planetary neighbor?": "Venus is Earth's closest planetary neighbor.",
    "What is the third brightest object in the sky after the Sun and the Moon?": "Venus is the third brightest object in the sky after the Sun and Moon.",
    "How does Venus spin compared to most planets?": "Venus spins slowly in the opposite direction from most planets.",
    "Why is Venus sometimes called Earth's evil twin?": "Venus is similar in structure and size to Earth, and its thick atmosphere makes it extremely hot, earning it the nickname 'Earth's evil twin.'",
    "What makes Venus the hottest planet in our solar system?": "Venus has a thick atmosphere that traps heat in a runaway greenhouse effect, making it the hottest planet in our solar system.",
    "What is Venus' average distance from the Sun?": "Venus orbits the Sun from an average distance of 67 million miles (108 million kilometers).",
    "How long does it take for sunlight to travel from the Sun to Venus?": "It takes sunlight about six minutes to travel from the Sun to Venus.",
    "How does Venus' size compare to Earth's?": "Venus' diameter is about 7,521 miles (12,104 kilometers), which is slightly smaller than Earth's diameter of 7,926 miles (12,756 kilometers).",
    "Why did the ancients think Venus was two different objects?": "Ancients thought Venus was two different objects, a morning star and an evening star, due to the way it appears in the sky at different times.",
    "What would it be like to spend a day on Venus?": "Spending a day on Venus would be disorienting, with extremely high temperatures around 900 degrees Fahrenheit (475 Celsius), and a day that lasts 243 Earth days.",
    "How long is a day on Venus compared to a Venusian year?": "A day on Venus lasts 243 Earth days, which is longer than a year on Venus, which takes only 225 Earth days.",
    "Why does the Sun rise in the west and set in the east on Venus?": "On Venus, the Sun rises in the west and sets in the east because Venus spins in the opposite direction compared to Earth.",
    "Why doesn't Venus have seasons like Earth?": "Venus doesn’t have noticeable seasons because its axis tilt is only three degrees, unlike Earth's 23-degree tilt.",
    "How do Venus and Earth compare in structure?": "Venus and Earth have similar structures, with iron cores surrounded by hot rock mantles and a thin rocky crust.",
    "What is plate tectonics and how does it relate to Venus?": "Plate tectonics is the process of continents moving and reshaping the surface of a planet. Venus may have experienced subduction, a key part of tectonics.",
    "How does Earth's tilt affect its seasons?": "Earth's axis is tilted 23.4 degrees, causing different parts of the planet to receive varying amounts of sunlight, which leads to the changing seasons.",
    "What are the main layers of Earth?": "Earth is composed of four main layers: the inner core, outer core, mantle, and crust.",
    "How long does it take for light from the Sun to reach Earth?": "It takes about eight minutes for sunlight to travel from the Sun to Earth.",
    "What is the equatorial diameter of Earth?": "The equatorial diameter of Earth is 7,926 miles (12,760 kilometers).",
    "Why is Mars one of the most explored bodies in our solar system?": "Mars is one of the most explored bodies because NASA has sent multiple missions and rovers to study its surface, finding evidence of a wetter and warmer past.",
    "Why is Mars called the 'Red Planet'?": "Mars is called the 'Red Planet' because its surface is covered in iron oxide, or rust, which gives it a reddish appearance.",
    "How many moons does Mars have?": "Mars has two small moons, Phobos and Deimos, which are likely captured asteroids.",
    "How far is Mars from the Sun?": "Mars is about 142 million miles (228 million kilometers) from the Sun, which is 1.5 astronomical units (AU).",
    "How long does it take sunlight to reach Mars?": "It takes sunlight approximately 13 minutes to travel from the Sun to Mars.",
    "What is the core of Mars made of?": "Mars has a dense core made of iron, nickel, and sulfur, with a radius between 930 and 1,300 miles (1,500 to 2,100 kilometers).",
    "What is Mars' atmosphere made of?": "Mars' atmosphere is composed mainly of carbon dioxide, nitrogen, and argon gases.",
    "What is the range of temperatures on Mars?": "Temperatures on Mars can range from 70 degrees Fahrenheit (20 degrees Celsius) during the day to -225 degrees Fahrenheit (-153 degrees Celsius) at night.",
    "Why does Mars' sky appear hazy and red?": "Mars' sky appears hazy and red due to the suspended dust in the thin atmosphere, which scatters light differently than Earth's atmosphere.",
    "What is the size of Mars compared to Earth?": "Mars has a radius of 2,106 miles (3,390 kilometers), which is about half the size of Earth.",
    "What are Jupiter's stripes and swirls made of?": "Jupiter's stripes and swirls are made of cold, windy clouds of ammonia and water, floating in an atmosphere of hydrogen and helium.",
    "What is the Great Red Spot on Jupiter?": "The Great Red Spot is a massive storm on Jupiter that is larger than Earth and has been active for hundreds of years.",
    "How many moons does Jupiter have?": "Jupiter has 95 moons that are officially recognized by the International Astronomical Union.",
    "What are Jupiter's four largest moons called?": "Jupiter's four largest moons are called the Galilean satellites: Io, Europa, Ganymede, and Callisto.",
    "How big is Jupiter compared to Earth?": "Jupiter is 11 times wider than Earth, with a radius of 43,440.7 miles (69,911 kilometers).",
    "How long does it take for Jupiter to orbit the Sun?": "Jupiter takes about 12 Earth years (4,333 Earth days) to make a complete orbit around the Sun.",
    "What is Jupiter's atmosphere primarily made of?": "Jupiter's atmosphere is mostly made of hydrogen and helium.",
    "How fast do winds on Jupiter blow?": "Winds on Jupiter can reach speeds of up to 335 miles per hour (539 kilometers per hour) at the equator.",
    "Why do Jupiter's bands and belts flow in opposite directions?": "Jupiter's fast rotation creates strong jet streams, causing the dark belts and light zones to flow east and west in opposite directions.",
    "How long does one day on Jupiter last?": "One day on Jupiter lasts about 10 Earth hours, which is the time it takes for Jupiter to complete one rotation.",
    "What position does Saturn hold in the solar system?": "Saturn is the sixth planet from the Sun.",
    "What is Saturn primarily made of?": "Saturn is made mostly of hydrogen and helium.",
    "How do Saturn's rings compare to those of other planets?": "Saturn's rings are the most spectacular and complex of any planet in the solar system.",
    "What is unique about Saturn's moon Titan?": "Saturn's moon Titan has methane lakes and is covered by a thick, smoggy atmosphere.",
    "How wide is Saturn compared to Earth?": "Saturn is about 9 times wider than Earth, with an equatorial diameter of 74,897 miles (120,500 kilometers).",
    "How far is Saturn from the Sun?": "Saturn is about 886 million miles (1.4 billion kilometers) from the Sun, or 9.5 astronomical units.",
    "How long is a day on Saturn?": "A day on Saturn lasts only 10.7 hours.",
    "How many moons does Saturn have?": "As of June 8, 2023, Saturn has 146 moons.",
    "What are Saturn's rings made of?": "Saturn's rings are made of billions of small chunks of ice and rock, coated with materials like dust.",
    "How fast are the winds in Saturn's upper atmosphere?": "Winds in Saturn's upper atmosphere can reach speeds of up to 1,600 feet per second (500 meters per second).",
    "What position does Saturn hold in the solar system?": "Saturn is the sixth planet from the Sun.",
    "What is Saturn primarily made of?": "Saturn is made mostly of hydrogen and helium.",
    "What is the position of Uranus in the solar system?": "Uranus is the seventh planet from the Sun.",
    "How does Uranus' rotation differ from other planets?": "Uranus appears to spin sideways, rotating at a nearly 90-degree angle from the plane of its orbit.",
    "How many rings and moons does Uranus have?": "Uranus is surrounded by 13 faint rings and 28 small moons.",
    "Who discovered Uranus and when?": "Uranus was discovered in 1781 by astronomer William Herschel.",
    "What is the equatorial diameter of Uranus compared to Earth?": "Uranus has an equatorial diameter of 31,763 miles (51,118 kilometers), which is four times wider than Earth.",
    "How far is Uranus from the Sun?": "Uranus is 1.8 billion miles (2.9 billion kilometers) from the Sun, or about 19 astronomical units.",
    "How long is a day on Uranus?": "A day on Uranus takes about 17 hours.",
    "How long does Uranus take to orbit the Sun?": "Uranus takes about 84 Earth years (30,687 Earth days) to orbit the Sun.",
    "What causes Uranus' extreme seasons?": "Uranus' unique tilt of 97.77 degrees causes it to have the most extreme seasons in the solar system.",
    "What is Uranus made of?": "Most of Uranus' mass is made up of a hot dense fluid of water, methane, and ammonia above a small rocky core.",
    "What is Neptune's position in the solar system?": "Neptune is the eighth and most distant planet in our solar system.",
    "How does Neptune's distance from the Sun compare to Earth's?": "Neptune is more than 30 times as far from the Sun as Earth.",
    "Why can't Neptune be seen with the naked eye?": "Neptune is the only planet in our solar system not visible to the naked eye due to its great distance from the Sun.",
    "How long does it take Neptune to complete one orbit around the Sun?": "Neptune completes one orbit around the Sun in about 165 Earth years.",
    "What is Neptune's equatorial diameter compared to Earth?": "Neptune has an equatorial diameter of 30,775 miles (49,528 kilometers), which is about four times wider than Earth.",
    "How far is Neptune from the Sun?": "Neptune is 2.8 billion miles (4.5 billion kilometers) from the Sun, or 30 astronomical units away.",
    "How long does it take for sunlight to reach Neptune?": "It takes sunlight about 4 hours to travel from the Sun to Neptune.",
    "What are the names of Neptune's rings?": "Neptune's rings are named Galle, Leverrier, Lassell, Arago, and Adams.",
    "What is Neptune primarily made of?": "Neptune is made of a hot dense fluid of icy materials like water, methane, and ammonia above a small, rocky core.",
    "How long is a day on Neptune?": "A day on Neptune takes about 16 hours.",
    "What are the three criteria that define a planet according to the IAU?": "A planet must orbit its host star, be mostly round, and have cleared its orbit of other objects of similar size.",
    "Why was Pluto reclassified as a dwarf planet?": "Pluto was reclassified as a dwarf planet in 2006 because it does not clear other objects of similar size from its orbit.",
    "Who discovered Pluto and when?": "Pluto was discovered by Clyde Tombaugh in 1930.",
    "What caused outrage when Pluto was reclassified as a dwarf planet?": "The reclassification of Pluto as a dwarf planet caused widespread outrage because it had long been considered the ninth planet.",
    "When did NASA’s New Horizons spacecraft make its historic flight through the Pluto system?": "NASA’s New Horizons spacecraft flew through the Pluto system on July 14, 2015.",
    "What did the New Horizons mission contribute to our understanding of Pluto?": "The New Horizons mission provided the first close-up images of Pluto and its moons, transforming our understanding of these distant worlds.",
    "What is Ceres, and where is it located?": "Ceres is the largest object in the asteroid belt between Mars and Jupiter, and it is the only dwarf planet located in the inner solar system.",
    "Was Ceres always considered a dwarf planet?": "No, like Pluto, Ceres was once classified as a planet before being reclassified as a dwarf planet.",
    "Which spacecraft was the first to visit Ceres?": "NASA’s Dawn mission was the first spacecraft to visit the dwarf planet Ceres.",
    "Where is Pluto located in the solar system?": "Pluto is located in the distant Kuiper Belt, beyond the orbit of Neptune.",
    "Why do all moons share the same name as our Moon?": "People didn't know other moons existed until Galileo Galilei discovered four moons orbiting Jupiter in 1610, so they called our Moon just 'the Moon.'",
    "What is the Latin name for the Moon, and how is it used?": "The Latin name for the Moon is 'Luna', and it is used as the adjective for Moon-related things, such as 'lunar.'",
    "How does the size of the Moon compare to Earth?": "The Moon's radius is about 1,080 miles (1,740 kilometers), which is less than a third of Earth's width.",
    "How far is the Moon from Earth?": "The Moon is an average of 238,855 miles (384,400 kilometers) away from Earth.",
    "How does the Moon's distance from Earth change over time?": "The Moon is slowly moving away from Earth, getting about an inch farther away each year.",
    "Why do we always see the same side of the Moon from Earth?": "The Moon rotates at the same rate that it revolves around Earth (synchronous rotation), so the same hemisphere always faces Earth.",
    "What causes the Moon's phases?": "The Moon's phases are caused by different parts of the Moon being illuminated by the Sun as it orbits Earth.",
    "How long does it take the Moon to orbit Earth?": "The Moon makes a complete orbit around Earth in 27 Earth days.",
    "How did the Moon form?": "The Moon likely formed after a Mars-sized body collided with Earth, and the resulting debris accumulated to form our natural satellite.",
    "What did NASA's SOFIA discover about the Moon in 2020?": "NASA's SOFIA confirmed the presence of water on the sunlit surface of the Moon in 2020, indicating water may be distributed across the lunar surface.",
    "What is a periodic comet?": "A periodic comet orbits the sun and returns to the inner solar system on a regular basis.",
    "How long does it take Halley's Comet to complete one orbit around the sun?": "It takes 75–76 years for Halley's Comet to orbit the sun and return to Earth.",
    "How large is Halley's Comet?": "Halley's Comet is about 11 kilometers in diameter, comparable in size to the city of Boston.",
    "What is Halley's Comet made of?": "Halley's Comet is made up of water and gases like methane, ammonia, and carbon dioxide.",
    "What happens to Halley's Comet as it approaches the sun?": "As Halley's Comet approaches the sun, the ice and gases inside it expand and form a tail.",
    "How many tails does Halley's Comet have?": "Halley's Comet has two tails: a dust tail and an ion (gas) tail.",
    "Who predicted the return of Halley's Comet?": "Edmond Halley predicted the return of Halley's Comet in 1758.",
    "When was it first proven that some comets are part of our solar system?": "It was first proven that some comets are part of our solar system when Edmond Halley showed that comets seen in 1531, 1607, and 1682 were actually the same comet.",
    "When will Halley's Comet next appear?": "Halley's Comet's next appearance is predicted to be in July 2061.",
    "How often can Halley's Comet be seen from Earth?": "Halley's Comet can be seen from Earth roughly every 75–76 years.",
    "How large is Apophis?": "Apophis is about 375 meters wide, roughly the size of a cruise liner.",
    "What is an Aten asteroid?": "An Aten asteroid has an orbit with a semi-major axis of less than one astronomical unit.",
    "How often does Apophis orbit the Sun?": "Apophis completes an orbit around the Sun every 323,513 days.",
    "When will Apophis make its close pass to Earth?": "On April 13, 2029, Apophis will pass within 32,000 kilometers of Earth's surface.",
    "How many people will be able to see Apophis when it passes by Earth in 2029?": "Apophis will be visible to about two billion people in parts of Asia, Europe, and Africa.",
    "How close will Apophis pass to Earth in 2036?": "In 2036, Apophis will pass more than 49 million kilometers from Earth.",
    "What is the closest Apophis could pass to Earth in 2116?": "Apophis could pass as close as 150,000 kilometers from Earth in 2116.",
    "Does Apophis pose any threat to Earth?": "NASA says that Apophis poses no threat to Earth for at least the next century.",
    "When was Apophis discovered?": "Apophis was discovered in 2004.",
    "What is the mythological origin of the name Apophis?": "Apophis is also known as Apep, the great serpent and enemy of the Egyptian sun god Ra.",
    "Where is the asteroid belt located in the solar system?": "The asteroid belt is located between the orbits of Mars and Jupiter.",
    "What shape does the asteroid belt take?": "The asteroid belt is a torus-shaped or disc-shaped region.",
    "What are asteroids in the asteroid belt made of?": "The asteroids are composed of carbon-rich materials, ices, and sometimes iron and nickel.",
    "How old is the asteroid belt?": "The asteroid belt formed about 4.5 billion years ago, along with the rest of the solar system.",
    "How many asteroids in the asteroid belt have been identified?": "Over 600,000 asteroids in the belt have been identified and named.",
    "What is the largest known asteroid in the asteroid belt?": "Ceres is the largest known asteroid, measuring 620 miles across.",
    "Is the asteroid belt densely packed?": "No, the asteroid belt is not as densely packed as often portrayed in fiction.",
    "What color is the asteroid belt?": "The asteroid belt is largely empty space, so there isn’t much color to see.",
    "What is the distance between the asteroid belt and the Sun?": "The asteroid belt exists between 330 million and 480 million kilometers from the Sun.",
    "What are resonances in the asteroid belt?": "The asteroid belt exhibits population depletions, or gaps, due to resonances in the mean distances of the asteroids."
    """
    
    "Sun name": "Sun",
    "Sun Vol. Mean Radius": "695700 km",
    "Sun Density": "1.408 g/cm³",
    "Sun Mass": "1.989 ×10³⁰ kg",
    "Sun Volume": "1.412 ×10¹⁰ km³",
    "Sun Sidereal rot. period": "25.38 días",
    "Sun Sid. rot. rate": "2.865×10⁻⁶ rad/s",
    "Sun Mean solar day": "24.47 días",
    "Sun Core radius": "~150000 km",
    "Sun Geometric Albedo": "N/A",
    "Sun GM": "132712440018 km³/s²",
    "Sun Equatorial radius": "695700 km",
    "Sun GM 1-sigma": "N/A",
    "Sun Mass ratio": "N/A",
    "Sun Moment of Inertia": "N/A",
    "Sun Equatorial gravity": "274.0 m/s²",
    "Sun Atmospheric pressure": "N/A",
    "Sun Max. angular diameter": "32.0 arcsec",
    "Sun Mean Temperature": "5778 K",
    "Sun Visual mag.": "-26.74",
    "Sun Obliquity to orbit": "7.25 degrees",
    "Sun Hill's sphere radius": "N/A",
    "Sun Sidereal orbital period": "N/A",
    "Sun Mean orbital velocity": "N/A",
    "Sun Escape velocity": "617.7 km/s",
    "Sun Solar Constant Perihelion": "1367 W/m²",
    "Sun Solar Constant Aphelion": "1322 W/m²",
    "Sun Solar Constant Mean": "1350 W/m²",

    "Mercury name": "Mercury",
    "Mercury Vol. Mean Radius": "2439.4±0.1 km",
    "Mercury Density": "5.427 g/cm³",
    "Mercury Mass": "3.302 ×10²³ kg",
    "Mercury Volume": "6.085 ×10¹⁰ km³",
    "Mercury Sidereal rot. period": "58.6463 days",
    "Mercury Sid. rot. rate": "0.00000124001 rad/s",
    "Mercury Mean solar day": "175.9421 days",
    "Mercury Core radius": "~1600 km",
    "Mercury Geometric Albedo": "0.106",
    "Mercury GM": "22031.86855 km³/s²",
    "Mercury Equatorial radius": "2440.53 km",
    "Mercury GM 1-sigma": "N/A",
    "Mercury Mass ratio": "6023682 Sun/planet",
    "Mercury Moment of Inertia": "0.33",
    "Mercury Equatorial gravity": "3.701 m/s²",
    "Mercury Atmospheric pressure": "<5×10⁻¹⁵ bar",
    "Mercury Max. angular diameter": "11.0 arcsec",
    "Mercury Mean Temperature": "440 K",
    "Mercury Visual mag.": "-0.42",
    "Mercury Obliquity to orbit": "2.11' ± 0.1'",
    "Mercury Hill's sphere radius": "94.4 Rp",
    "Mercury Sidereal orbital period": "0.2408467 years",
    "Mercury Mean orbital velocity": "47.362 km/s",
    "Mercury Sidereal orbital period (days)": "87.969257 days",
    "Mercury Escape velocity": "4.435 km/s",
    "Mercury Solar Constant Perihelion": "14462 W/m²",
    "Mercury Solar Constant Aphelion": "6278 W/m²",
    "Mercury Solar Constant Mean": "9126 W/m²",

    "Mars name": "Mars",
    "Mars mass": "6.4171 x10^23 kg ",
    "Mars Vol. Mean Radius": "3389.92+-0.04 km",
    "Mars Density": "3.933 5+-4 g cm^-3",
    "Mars Volume": "16.318 x10^10 km^3",
    "Mars Sidereal rot. period": "24.622962 hr",
    "Mars Sid. rot. rate": "0.0000708822 rad/s",
    "Mars Mean solar day": "88775.24415 s",
    "Mars Core radius": "~1700 km",
    "Mars Geometric Albedo": "0.150",
    "Mars GM": "42828.375214 km^3/s^2",
    "Mars Equatorial radius": "3396.19 km",
    "Mars GM 1-sigma": "+- 0.00028 km^3/s^2",
    "Mars Mass ratio": "3098703.59 Sun/planet",
    "Mars Moment of Inertia": "N/A",
    "Mars Equatorial gravity": "3.71 m/s^2",
    "Mars Atmospheric pressure": "0.0056 bar",
    "Mars Max. angular diameter": "17.9 arcsec",
    "Mars Mean Temperature": "210 K",
    "Mars Visual mag.": "-1.52",
    "Mars Obliquity to orbit": "25.19 deg",
    "Mars Hill's sphere radius": "319.8 Rp",
    "Mars Sidereal orbital period": "1.88081578 y",
    "Mars Mean orbital velocity": "24.13 km/s",
    "Mars Sidereal orbital period (days)": "686.98",
    "Mars Escape velocity": "5.027 km/s",
    "Mars Solar Constant Perithelion": "717 W/m^2",
    "Mars Solar Constant Aphelion": "493 W/m^2",
    "Mars Solar Constant Mean": "589 W/m^2",
    "Mars Maximum Planetary IR Perihelion": "470 W/m^2",
    "Mars Maximum Planetary IR Aphelion": "315 W/m^2",
    "Mars Maximum Planetary IR Mean": "390 W/m^2",
    "Mars Minimum Planetary IR Perihelion": "30 W/m^2",
    "Mars Minimum Planetary IR Aphelion": "30 W/m^2",
    "Mars Minimum Planetary IR Mean": "30 W/m^2",

    "Venus name": "Venus", 
    "Venus Vol. Mean Radius": "6051.8+-0.1 km",
    "Venus Density": "5.204 g cm^-3",
    "Venus Mass": "48.685 x10^23 kg",
    "Venus Volume": "92.843 x10^10 km^3",
    "Venus Sidereal rot. period": "243.025 d",
    "Venus Sid. rot. rate": "-0.00029924 rad/s",
    "Venus Mean solar day": "116.75 d",
    "Venus Core radius": "~3000 km",
    "Venus Geometric Albedo": "0.65",
    "Venus GM": "324858.599 km^3/s^2",
    "Venus Equatorial radius": "6051.8 km",
    "Venus GM 1-sigma": "",
    "Venus Mass ratio": "408523.71 Sun/planet",
    "Venus Moment of Inertia": "0.33",
    "Venus Equatorial gravity": "8.87 m/s^2",
    "Venus Atmospheric pressure": "92.1 bar",
    "Venus Max. angular diameter": "66 arcsec",
    "Venus Mean Temperature": "737 K",
    "Venus Visual mag.": "-4.47",
    "Venus Obliquity to orbit": "177.4' +/- 0.1'",
    "Venus Hill's sphere radius": "162 Rp",
    "Venus Sidereal orbital period": "0.61519726 y",
    "Venus Mean orbital velocity": "35.02 km/s",
    "Venus Sidereal orbital period (days)": "224.701",
    "Venus Escape velocity": "10.36 km/s",
    "Venus Solar Constant Perithelion": "2614 W/m^2",
    "Venus Solar Constant Aphelion": "2608 W/m^2",
    "Venus Solar Constant Mean": "2611 W/m^2",

    "Jupiter name": "Jupiter",
    "Jupiter Vol. Mean Radius": "69911+-6 km",
    "Jupiter Density": "1.3262 +- .0003 g/cm^3",
    "Jupiter Mass": "189818722 +- 8817 x10^22 g",
    "Jupiter Equatorial radius": "71492+-4 km",
    "Jupiter Polar radius": "66854+-10 km",
    "Jupiter Flattening": "0.06487",
    "Jupiter Geometric Albedo": "0.52",
    "Jupiter Rocky core mass": "0.0261",
    "Jupiter Sidereal rot. period": "9h 55m 29.711 s",
    "Jupiter Sid. rot. rate": "0.00017585 rad/s",
    "Jupiter Mean solar day": "~9.9259 hr",
    "Jupiter GM": "126686531.900 km^3/s^2",
    "Jupiter GM 1-sigma": "+- 1.2732 km^3/s^2",
    "Jupiter Equatorial gravity": "24.79 m/s^2",
    "Jupiter Polar gravity": "28.34 m/s^2",
    "Jupiter Visual mag.": "-9.40",
    "Jupiter Visual mag. opposition": "-2.70",
    "Jupiter Obliquity to orbit": "3.13 deg",
    "Jupiter Sidereal orbital period": "11.861982204 y",
    "Jupiter Sidereal orbital period days": "4332.589",
    "Jupiter Mean orbital speed": "13.0697 km/s",
    "Jupiter Atmospheric temperature": "165+-5 K",
    "Jupiter Escape velocity": "59.5 km/s",

    "Saturn name": "Saturn",
    "Saturn Vol. Mean Radius": "58232 +- 6 km",
    "Saturn Density": "0.687+-0.001 g cm^-3",
    "Saturn Mass": "5.6834 x10^26 kg",
    "Saturn Volume": "N/A",
    "Saturn Sidereal rot. period": "10h 39m 22.4s",
    "Saturn Sid. rot. rate": "0.000163785 rad/s",
    "Saturn Mean solar day": "10.656 hrs",
    "Saturn Core radius": "N/A",
    "Saturn Geometric Albedo": "0.47",
    "Saturn GM": "37931206.234 km^3/s^2",
    "Saturn Equatorial radius": "60268 +- 4 km",
    "Saturn GM 1-sigma": "N/A",
    "Saturn Mass ratio": "N/A",
    "Saturn Moment of Inertia": "N/A",
    "Saturn Equatorial gravity": "10.44 m/s^2",
    "Saturn Atmospheric pressure": "N/A",
    "Saturn Max. angular diameter": "N/A",
    "Saturn Mean Temperature": "N/A",
    "Saturn Visual mag.": "-8.88",
    "Saturn Obliquity to orbit": "26.73 deg",
    "Saturn Hill's sphere radius": "1100 Rp",
    "Saturn Sidereal orbital period": "29.447498 y",
    "Saturn Mean orbital velocity": "9.68 km/s",
    "Saturn Sidereal orbital period days": "10755.698",
    "Saturn Escape velocity": "35.5 km/s",
    "Saturn Solar Constant Perithelion": "16.8 W/m^2",
    "Saturn Solar Constant Aphelion": "13.6 W/m^2",
    "Saturn Solar Constant Mean": "15.1 W/m^2",
    "Saturn Maximum Planetary IR Perihelion": "4.7 W/m^2",
    "Saturn Maximum Planetary IR Aphelion": "4.5 W/m^2",
    "Saturn Maximum Planetary IR Mean": "4.6 W/m^2",
    "Saturn Minimum Planetary IR Perihelion": "4.7 W/m^2",
    "Saturn Minimum Planetary IR Aphelion": "4.5 W/m^2",
    "Saturn Minimum Planetary IR Mean": "4.6 W/m^2",

    "Uranus name": "Uranus",
    "Uranus Vol. Mean Radius": "25362 +- 12 km",
    "Uranus Density": "1.271 g cm^-3",
    "Uranus Mass": "86.813 x10^24 kg",
    "Uranus Volume": "N/A",
    "Uranus Sidereal rot. period": "17.24 +- 0.01 hr",
    "Uranus Sid. rot. rate": "-0.000101237 rad/s",
    "Uranus Mean solar day": "17.24 h",
    "Uranus Core radius": "N/A",
    "Uranus Geometric Albedo": "0.51",
    "Uranus GM": "5793951.256 km^3/s^2",
    "Uranus Equatorial radius": "25559 +- 4 km",
    "Uranus GM 1-sigma": "N/A",
    "Uranus Mass ratio": "N/A",
    "Uranus Moment of Inertia": "N/A",
    "Uranus Equatorial gravity": "8.87 m/s^2",
    "Uranus Atmospheric pressure": "N/A",
    "Uranus Max. angular diameter": "N/A",
    "Uranus Mean Temperature": "N/A",
    "Uranus Visual mag.": "-7.11",
    "Uranus Obliquity to orbit": "97.77 deg",
    "Uranus Hill's sphere radius": "2700 Rp",
    "Uranus Sidereal orbital period": "84.0120465 y",
    "Uranus Mean orbital velocity": "6.8 km/s",
    "Uranus Sidereal orbital period days": "30685.4",
    "Uranus Escape velocity": "21.3 km/s",
    "Uranus Solar Constant Perithelion": "4.09 W/m^2",
    "Uranus Solar Constant Aphelion": "3.39 W/m^2",
    "Uranus Solar Constant Mean": "3.71 W/m^2",
    "Uranus Maximum Planetary IR Perihelion": "0.72 W/m^2",
    "Uranus Maximum Planetary IR Aphelion": "0.55 W/m^2",
    "Uranus Maximum Planetary IR Mean": "0.63 W/m^2",
    "Uranus Minimum Planetary IR Perihelion": "0.72 W/m^2",
    "Uranus Minimum Planetary IR Aphelion": "0.55 W/m^2",
    "Uranus Minimum Planetary IR Mean": "0.63 W/m^2",

    "Neptune name": "Neptune",
    "Neptune Vol. Mean Radius": "24624+-21 km",
    "Neptune Density": "1.638 g cm^-3",
    "Neptune Mass": "102.409 kg",
    "Neptune Volume": "6254 x10^10 km^3",
    "Neptune Sidereal rot. period": "16.11+-0.01 hr",
    "Neptune Sid. rot. rate": "0.000108338 rad/s",
    "Neptune Mean solar day": "~16.11 hr",
    "Neptune Core radius": "24342+-30 km",
    "Neptune Geometric Albedo": "0.41",
    "Neptune GM": "6835099.97 km^3/s^2",
    "Neptune Equatorial radius": "24766+-15 km",
    "Neptune GM 1-sigma": "+-10 km^3/s^2",
    "Neptune Mass ratio": "N/A",
    "Neptune Moment of Inertia": "N/A",
    "Neptune Equatorial gravity": "11.15 m/s^2",
    "Neptune Atmospheric pressure": "N/A bar",
    "Neptune Max. angular diameter": "N/A arcsec",
    "Neptune Mean Temperature": "72+-2 K",
    "Neptune Visual mag.": "-6.87",
    "Neptune Obliquity to orbit": "28.32 deg",
    "Neptune Hill's sphere radius": "4700 Rp",
    "Neptune Sidereal orbital period": "164.788501027 y",
    "Neptune Mean orbital velocity": "5.43 km/s",
    "Neptune Sidereal orbital period days": "60189",
    "Neptune Escape velocity": "23.5 km/s",
    "Neptune Solar Constant Perihelion": "1.54 W/m^2",
    "Neptune Solar Constant Aphelion": "1.49 W/m^2",
    "Neptune Solar Constant Mean": "1.51 W/m^2",
    "Neptune Maximum Planetary IR Perihelion": "0.52 W/m^2",
    "Neptune Maximum Planetary IR Aphelion": "0.52 W/m^2",
    "Neptune Maximum Planetary IR Mean": "0.52 W/m^2",
    "Neptune Minimum Planetary IR Perihelion": "0.52 W/m^2",
    "Neptune Minimum Planetary IR Aphelion": "0.52 W/m^2",
    "Neptune Minimum Planetary IR Mean": "0.52 W/m^2",

    "Pluto name": "Pluto",
    "Pluto Vol. Mean Radius": "1188.3+-1.6 km",
    "Pluto Density": "1.86 g cm^-3",
    "Pluto Mass": "1.307+-0.018 kg",
    "Pluto Volume": "0.697 x10^10 km^3",
    "Pluto Sidereal rot. period": "153.29335198 h",
    "Pluto Sid. rot. rate": "0.0000113856 rad/s",
    "Pluto Mean solar day": "153.2820 h",
    "Pluto Core radius": "N/A",
    "Pluto Geometric Albedo": "N/A",
    "Pluto GM": "869.326 km^3/s^2",
    "Pluto Equatorial radius": "1188.3 km",
    "Pluto GM 1-sigma": "0.4 km^3/s^2",
    "Pluto Mass ratio": "N/A",
    "Pluto Moment of Inertia": "N/A",
    "Pluto Equatorial gravity": "0.611 m/s^2",
    "Pluto Atmospheric pressure": "N/A bar",
    "Pluto Max. angular diameter": "N/A arcsec",
    "Pluto Mean Temperature": "N/A K",
    "Pluto Visual mag.": "N/A",
    "Pluto Obliquity to orbit": "N/A",
    "Pluto Hill's sphere radius": "N/A Rp",
    "Pluto Sidereal orbital period": "249.58932 y",
    "Pluto Mean orbital velocity": "4.67 km/s",
    "Pluto Sidereal orbital period days": "N/A",
    "Pluto Escape velocity": "1.21 km/s",
    "Pluto Solar Constant Perihelion": "1.56 W/m^2",
    "Pluto Solar Constant Aphelion": "0.56 W/m^2",
    "Pluto Solar Constant Mean": "0.88 W/m^2",
    "Pluto Maximum Planetary IR Perihelion": "0.8 W/m^2",
    "Pluto Maximum Planetary IR Aphelion": "0.3 W/m^2",
    "Pluto Maximum Planetary IR Mean": "0.5 W/m^2",
    "Pluto Minimum Planetary IR Perihelion": "0.8 W/m^2",
    "Pluto Minimum Planetary IR Aphelion": "0.3 W/m^2",
    "Pluto Minimum Planetary IR Mean": "0.5 W/m^2",
        
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

# Precomputar las incrustaciones para todos los contextos
context_descriptions = list(contexts.keys())
context_texts = list(contexts.values())
context_embeddings = embedding_model.encode(context_descriptions, convert_to_tensor=True)

@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    question = data.get('message', '')

    # Procesar la pregunta
    question_embedding = embedding_model.encode(question, convert_to_tensor=True)
    similarities = util.pytorch_cos_sim(question_embedding, context_embeddings)
    best_match_idx = torch.argmax(similarities)
    best_context_description = context_descriptions[best_match_idx]
    best_context_text = contexts[best_context_description]
    qa_input = {
        "question": question,
        "context": best_context_text
    }
    result = nlp(qa_input, max_answer_len=100)

    answer = result['answer']

    # Devolver la respuesta en formato JSON
    return jsonify({'response': answer})

if __name__ == '__main__':
    app.run(port=5000)
