from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline
from sentence_transformers import SentenceTransformer, util
import torch

# Load models
model_name = "distilbert/distilbert-base-cased-distilled-squad"
nlp = pipeline('question-answering', model=model_name, tokenizer=model_name)
embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# Predefined context with corresponding descriptions
contexts = {
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
