"""
Prompt Generator for Mamba-Friendly Test Cases
Generates 100 prompts per level optimized for state-space models (Mamba, DenseMamba, etc.)
Mamba models work better with:
- Simple, direct questions
- Sequential information
- Clear patterns
- Repetitive structures
- Short contexts

STRUCTURE:
==========
Each level contains:
- ORIGINAL PROMPTS: From the original test suite (marked with is_original: True)
- NEW PROMPTS: Generated prompts optimized for Mamba (marked with is_original: False)

Level 1: 3 original + 97 new = 100 total
Level 2: 4 original + 96 new = 100 total
Level 3: 3 original + 97 new = 100 total
Level 4: 3 original + 97 new = 100 total
Level 5: 3 original + 97 new = 100 total
Level 6: 2 original + 98 new = 100 total
"""

import random
from typing import List, Dict

def generate_mamba_friendly_prompts() -> Dict[str, List[Dict]]:
    """
    Generate 100 prompts for each level, optimized for Mamba models.
    Returns prompts in the same format as get_progressive_test_suite().
    """
    
    # ============================================================
    # LEVEL 1: SIMPLE RECALL (100 prompts)
    # ============================================================
    level1_prompts = []
    
    # ============================================================
    # ORIGINAL PROMPTS (3) - From original test suite
    # ============================================================
    level1_prompts.append({
        'prompt': 'Question: What is my name?\nAnswer: My name is Alice.\nQuestion: What is my name?\nAnswer:',
        'expected': 'Alice',
        'alternatives': ['Alice', 'alice', 'My name is Alice'],
        'is_original': True,
    })
    level1_prompts.append({
        'prompt': 'Question: What is the code?\nAnswer: The code is BLUE42.\nQuestion: What is the code?\nAnswer:',
        'expected': 'BLUE42',
        'alternatives': ['BLUE42', 'blue42', 'The code is BLUE42'],
        'is_original': True,
    })
    level1_prompts.append({
        'prompt': 'Question: What is 2+2?\nAnswer: 2+2 equals 4.\nQuestion: What is 2+2?\nAnswer:',
        'expected': '4',
        'alternatives': ['4', 'four', '2+2 equals 4'],
        'is_original': True,
    })
    
    # ============================================================
    # NEW PROMPTS (97) - Generated for Mamba-friendly testing
    # ============================================================
    names = ['Bob', 'Carol', 'David', 'Emma', 'Frank', 'Grace', 'Henry', 'Ivy', 'Jack', 'Kate']
    colors = ['red', 'blue', 'green', 'yellow', 'purple', 'orange', 'pink', 'brown', 'black', 'white']
    numbers = list(range(1, 50))
    codes = ['ABC123', 'XYZ789', 'DEF456', 'GHI321', 'JKL654', 'MNO987', 'PQR147', 'STU258', 'VWX369', 'YZA741']
    items = ['apple', 'book', 'car', 'dog', 'egg', 'fish', 'guitar', 'hat', 'ice', 'jacket']
    
    for i in range(97):
        pattern = i % 10
        if pattern == 0:  # Name recall
            name = random.choice(names)
            level1_prompts.append({
                'prompt': f'Question: What is my name?\nAnswer: My name is {name}.\nQuestion: What is my name?\nAnswer:',
                'expected': name,
                'alternatives': [name, name.lower()],
                'is_original': False,
            })
        elif pattern == 1:  # Code recall
            code = random.choice(codes)
            level1_prompts.append({
                'prompt': f'Question: What is the code?\nAnswer: The code is {code}.\nQuestion: What is the code?\nAnswer:',
                'expected': code,
                'alternatives': [code, code.lower()],
                'is_original': False,
            })
        elif pattern == 2:  # Simple arithmetic
            a, b = random.sample(numbers[:20], 2)
            level1_prompts.append({
                'prompt': f'Question: What is {a}+{b}?\nAnswer: {a}+{b} equals {a+b}.\nQuestion: What is {a}+{b}?\nAnswer:',
                'expected': str(a+b),
                'alternatives': [str(a+b), f'{a}+{b} equals {a+b}'],
                'is_original': False,
            })
        elif pattern == 3:  # Color recall
            color = random.choice(colors)
            level1_prompts.append({
                'prompt': f'Question: What is the color?\nAnswer: The color is {color}.\nQuestion: What is the color?\nAnswer:',
                'expected': color,
                'alternatives': [color, color.capitalize()],
                'is_original': False,
            })
        elif pattern == 4:  # Item recall
            item = random.choice(items)
            level1_prompts.append({
                'prompt': f'Question: What is the item?\nAnswer: The item is {item}.\nQuestion: What is the item?\nAnswer:',
                'expected': item,
                'alternatives': [item, item.capitalize()],
                'is_original': False,
            })
        elif pattern == 5:  # Number recall
            num = random.choice(numbers[:30])
            level1_prompts.append({
                'prompt': f'Question: What is the number?\nAnswer: The number is {num}.\nQuestion: What is the number?\nAnswer:',
                'expected': str(num),
                'alternatives': [str(num)],
                'is_original': False,
            })
        elif pattern == 6:  # Age recall
            age = random.choice(numbers[10:50])
            name = random.choice(names)
            level1_prompts.append({
                'prompt': f'Question: How old is {name}?\nAnswer: {name} is {age} years old.\nQuestion: How old is {name}?\nAnswer:',
                'expected': str(age),
                'alternatives': [str(age), f'{age} years old'],
                'is_original': False,
            })
        elif pattern == 7:  # City recall
            cities = ['Paris', 'London', 'Tokyo', 'New York', 'Berlin', 'Madrid', 'Rome', 'Vienna', 'Moscow', 'Sydney']
            city = random.choice(cities)
            level1_prompts.append({
                'prompt': f'Question: What is the city?\nAnswer: The city is {city}.\nQuestion: What is the city?\nAnswer:',
                'expected': city,
                'alternatives': [city, city.lower()],
                'is_original': False,
            })
        elif pattern == 8:  # Animal recall
            animals = ['cat', 'dog', 'bird', 'fish', 'rabbit', 'hamster', 'turtle', 'snake', 'lizard', 'frog']
            animal = random.choice(animals)
            level1_prompts.append({
                'prompt': f'Question: What is the animal?\nAnswer: The animal is {animal}.\nQuestion: What is the animal?\nAnswer:',
                'expected': animal,
                'alternatives': [animal, animal.capitalize()],
                'is_original': False,
            })
        else:  # pattern == 9: Simple subtraction
            a, b = random.sample(numbers[:20], 2)
            if a < b:
                a, b = b, a
            level1_prompts.append({
                'prompt': f'Question: What is {a}-{b}?\nAnswer: {a}-{b} equals {a-b}.\nQuestion: What is {a}-{b}?\nAnswer:',
                'expected': str(a-b),
                'alternatives': [str(a-b), f'{a}-{b} equals {a-b}'],
                'is_original': False,
            })
    
    # ============================================================
    # LEVEL 2: TWO-HOP REASONING (100 prompts)
    # ============================================================
    level2_prompts = []
    
    # ============================================================
    # ORIGINAL PROMPTS (4) - From original test suite
    # ============================================================
    level2_prompts.append({
        'prompt': 'Question: Who is taller?\nFacts: Alice is taller than Bob. Bob is taller than Carol.\nQuestion: Who is the tallest?\nAnswer:',
        'expected': 'Alice',
        'alternatives': ['Alice', 'alice'],
        'is_original': True,
    })
    level2_prompts.append({
        'prompt': 'Question: What happens to the ground?\nFacts: If it rains, the ground gets wet. It is raining.\nQuestion: What happens to the ground?\nAnswer:',
        'expected': 'wet',
        'alternatives': ['wet', 'gets wet', 'the ground gets wet'],
        'is_original': True,
    })
    level2_prompts.append({
        'prompt': 'Question: What color is the car?\nFacts: Alice drives a red car. Bob drives Alice to work.\nQuestion: What color is the car Bob drives?\nAnswer:',
        'expected': 'red',
        'alternatives': ['red', 'Red'],
        'is_original': True,
    })
    level2_prompts.append({
        'prompt': 'Question: How much total?\nFacts: Apple costs 2 dollars. Orange costs 3 dollars.\nQuestion: If I buy one apple and one orange, how much total?\nAnswer:',
        'expected': '5',
        'alternatives': ['5', 'five', '5 dollars', '$5'],
        'is_original': True,
    })
    
    # ============================================================
    # NEW PROMPTS (96) - Generated for Mamba-friendly testing
    # ============================================================
    for i in range(96):
        pattern = i % 8
        if pattern == 0:  # Transitive comparison (taller/shorter)
            names_list = random.sample(names, 3)
            level2_prompts.append({
                'prompt': f'Question: Who is tallest?\nFacts: {names_list[0]} is taller than {names_list[1]}. {names_list[1]} is taller than {names_list[2]}.\nQuestion: Who is the tallest?\nAnswer:',
                'expected': names_list[0],
                'alternatives': [names_list[0], names_list[0].lower()],
                'is_original': False,
            })
        elif pattern == 1:  # Simple conditional logic
            conditions = [
                ('If it snows, the road gets icy. It is snowing.', 'icy'),
                ('If the sun shines, the day is bright. The sun is shining.', 'bright'),
                ('If you study, you learn. You are studying.', 'learn'),
                ('If you run, you get tired. You are running.', 'tired'),
            ]
            condition, result = random.choice(conditions)
            level2_prompts.append({
                'prompt': f'Question: What happens?\nFacts: {condition}\nQuestion: What happens?\nAnswer:',
                'expected': result,
                'alternatives': [result, f'gets {result}'],
                'is_original': False,
            })
        elif pattern == 2:  # Indirect reference (car color)
            name1, name2 = random.sample(names, 2)
            color = random.choice(colors)
            level2_prompts.append({
                'prompt': f'Question: What color is the car?\nFacts: {name1} drives a {color} car. {name2} drives {name1} to work.\nQuestion: What color is the car {name2} drives?\nAnswer:',
                'expected': color,
                'alternatives': [color, color.capitalize()],
                'is_original': False,
            })
        elif pattern == 3:  # Simple addition
            a, b = random.sample(numbers[:15], 2)
            items_list = ['apple', 'banana', 'orange', 'grape', 'pear']
            item1, item2 = random.sample(items_list, 2)
            level2_prompts.append({
                'prompt': f'Question: How much total?\nFacts: {item1.capitalize()} costs {a} dollars. {item2.capitalize()} costs {b} dollars.\nQuestion: If I buy one {item1} and one {item2}, how much total?\nAnswer:',
                'expected': str(a+b),
                'alternatives': [str(a+b), f'{a+b} dollars', f'${a+b}'],
                'is_original': False,
            })
        elif pattern == 4:  # Age comparison
            name1, name2, name3 = random.sample(names, 3)
            age1, age2, age3 = sorted(random.sample(numbers[20:50], 3), reverse=True)
            level2_prompts.append({
                'prompt': f'Question: Who is oldest?\nFacts: {name1} is {age1} years old. {name2} is {age2} years old. {name3} is {age3} years old.\nQuestion: Who is the oldest?\nAnswer:',
                'expected': name1,
                'alternatives': [name1, name1.lower()],
                'is_original': False,
            })
        elif pattern == 5:  # Location chain
            locations = ['house', 'school', 'park', 'store', 'library', 'hospital', 'restaurant', 'office']
            loc1, loc2 = random.sample(locations, 2)
            item = random.choice(items)
            level2_prompts.append({
                'prompt': f'Question: Where is the {item}?\nFacts: The {item} is in the {loc1}. The {loc1} is next to the {loc2}.\nQuestion: Is the {item} near the {loc2}?\nAnswer:',
                'expected': 'yes',
                'alternatives': ['yes', 'Yes', 'YES'],
                'is_original': False,
            })
        elif pattern == 6:  # Ownership transfer
            name1, name2 = random.sample(names, 2)
            item = random.choice(items)
            level2_prompts.append({
                'prompt': f'Question: Who owns the {item}?\nFacts: {name1} owns a {item}. {name2} borrows the {item} from {name1}.\nQuestion: Who has the {item} now?\nAnswer:',
                'expected': name2,
                'alternatives': [name2, name2.lower()],
                'is_original': False,
            })
        else:  # pattern == 7: Simple multiplication
            a, b = random.sample(numbers[:10], 2)
            item = random.choice(items)
            level2_prompts.append({
                'prompt': f'Question: How much total?\nFacts: One {item} costs {a} dollars. I buy {b} {item}s.\nQuestion: How much do I pay in total?\nAnswer:',
                'expected': str(a*b),
                'alternatives': [str(a*b), f'{a*b} dollars', f'${a*b}'],
                'is_original': False,
            })
    
    # ============================================================
    # LEVEL 3: THREE-HOP REASONING (100 prompts)
    # ============================================================
    level3_prompts = []
    
    # ============================================================
    # ORIGINAL PROMPTS (3) - From original test suite
    # ============================================================
    level3_prompts.append({
        'prompt': 'Question: Who is the shortest?\nFacts: Tom is taller than Jim. Jim is taller than Bob. Bob is taller than Sam.\nQuestion: Who is the shortest person?\nAnswer:',
        'expected': 'Sam',
        'alternatives': ['Sam', 'sam'],
        'is_original': True,
    })
    level3_prompts.append({
        'prompt': 'Question: What is Rex?\nFacts: All dogs are animals. All animals need food. Rex is a dog.\nQuestion: Does Rex need food?\nAnswer:',
        'expected': 'yes',
        'alternatives': ['yes', 'Yes', 'YES', 'Rex needs food'],
        'is_original': True,
    })
    level3_prompts.append({
        'prompt': 'Question: Where is the book?\nFacts: The book is on the table. The table is in the kitchen. The kitchen is in the house.\nQuestion: Is the book in the house?\nAnswer:',
        'expected': 'yes',
        'alternatives': ['yes', 'Yes', 'YES'],
        'is_original': True,
    })
    
    # ============================================================
    # NEW PROMPTS (97) - Generated for Mamba-friendly testing
    # ============================================================
    for i in range(97):
        pattern = i % 10
        if pattern == 0:  # Multi-step height comparison
            name_list = random.sample(names, 4)
            level3_prompts.append({
                'prompt': f'Question: Who is shortest?\nFacts: {name_list[0]} is taller than {name_list[1]}. {name_list[1]} is taller than {name_list[2]}. {name_list[2]} is taller than {name_list[3]}.\nQuestion: Who is the shortest?\nAnswer:',
                'expected': name_list[3],
                'alternatives': [name_list[3], name_list[3].lower()],
                'is_original': False,
            })
        elif pattern == 1:  # Syllogistic reasoning
            categories = [
                ('dogs', 'animals', 'need food'),
                ('birds', 'animals', 'can fly'),
                ('fish', 'animals', 'live in water'),
                ('trees', 'plants', 'need sunlight'),
                ('cars', 'vehicles', 'need fuel'),
            ]
            cat1, cat2, property_val = random.choice(categories)
            names_list = ['Rex', 'Max', 'Buddy', 'Luna', 'Charlie', 'Bella', 'Duke', 'Molly']
            name = random.choice(names_list)
            level3_prompts.append({
                'prompt': f'Question: What is {name}?\nFacts: All {cat1} are {cat2}. All {cat2} {property_val}. {name} is a {cat1[:-1]}.\nQuestion: Does {name} {property_val}?\nAnswer:',
                'expected': 'yes',
                'alternatives': ['yes', 'Yes', 'YES'],
                'is_original': False,
            })
        elif pattern == 2:  # Spatial reasoning chain
            locations = ['table', 'desk', 'shelf', 'drawer', 'box', 'bag', 'room', 'cabinet']
            loc1, loc2, loc3 = random.sample(locations, 3)
            item = random.choice(items)
            level3_prompts.append({
                'prompt': f'Question: Where is the {item}?\nFacts: The {item} is on the {loc1}. The {loc1} is in the {loc2}. The {loc2} is in the {loc3}.\nQuestion: Is the {item} in the {loc3}?\nAnswer:',
                'expected': 'yes',
                'alternatives': ['yes', 'Yes', 'YES'],
                'is_original': False,
            })
        elif pattern == 3:  # Age chain
            name_list = random.sample(names, 4)
            ages = sorted(random.sample(numbers[20:50], 4), reverse=True)
            level3_prompts.append({
                'prompt': f'Question: Who is oldest?\nFacts: {name_list[0]} is older than {name_list[1]}. {name_list[1]} is older than {name_list[2]}. {name_list[2]} is older than {name_list[3]}.\nQuestion: Who is the oldest?\nAnswer:',
                'expected': name_list[0],
                'alternatives': [name_list[0], name_list[0].lower()],
                'is_original': False,
            })
        elif pattern == 4:  # Color inheritance
            name1, name2, name3 = random.sample(names, 3)
            color = random.choice(colors)
            level3_prompts.append({
                'prompt': f'Question: What color?\nFacts: {name1} has a {color} shirt. {name2} borrows the shirt from {name1}. {name3} sees {name2}.\nQuestion: What color shirt does {name3} see?\nAnswer:',
                'expected': color,
                'alternatives': [color, color.capitalize()],
                'is_original': False,
            })
        elif pattern == 5:  # Multi-step arithmetic
            a, b, c = random.sample(numbers[:10], 3)
            level3_prompts.append({
                'prompt': f'Question: How much total?\nFacts: Apple costs {a} dollars. Orange costs {b} dollars. Banana costs {c} dollars.\nQuestion: If I buy one of each, how much total?\nAnswer:',
                'expected': str(a+b+c),
                'alternatives': [str(a+b+c), f'{a+b+c} dollars'],
                'is_original': False,
            })
        elif pattern == 6:  # Ownership chain
            name_list = random.sample(names, 3)
            item = random.choice(items)
            level3_prompts.append({
                'prompt': f'Question: Who has the {item}?\nFacts: {name_list[0]} owns a {item}. {name_list[1]} borrows it from {name_list[0]}. {name_list[2]} takes it from {name_list[1]}.\nQuestion: Who has the {item} now?\nAnswer:',
                'expected': name_list[2],
                'alternatives': [name_list[2], name_list[2].lower()],
                'is_original': False,
            })
        elif pattern == 7:  # Size comparison chain
            items_list = ['box', 'bag', 'jar', 'cup', 'bowl']
            item1, item2, item3, item4 = random.sample(items_list, 4)
            level3_prompts.append({
                'prompt': f'Question: What is smallest?\nFacts: The {item1} is bigger than the {item2}. The {item2} is bigger than the {item3}. The {item3} is bigger than the {item4}.\nQuestion: What is the smallest?\nAnswer:',
                'expected': item4,
                'alternatives': [item4, f'the {item4}'],
                'is_original': False,
            })
        elif pattern == 8:  # Time sequence
            activities = ['breakfast', 'lunch', 'dinner', 'snack', 'dessert']
            act1, act2, act3 = random.sample(activities, 3)
            level3_prompts.append({
                'prompt': f'Question: What comes first?\nFacts: {act1.capitalize()} comes before {act2}. {act2} comes before {act3}.\nQuestion: What comes first: {act1} or {act3}?\nAnswer:',
                'expected': act1,
                'alternatives': [act1, act1.capitalize()],
                'is_original': False,
            })
        else:  # pattern == 9: Category membership
            animals = ['dog', 'cat', 'bird', 'fish']
            animal = random.choice(animals)
            level3_prompts.append({
                'prompt': f'Question: What is {animal}?\nFacts: All {animal}s are pets. All pets need care. {animal.capitalize()}s are {animal}s.\nQuestion: Do {animal}s need care?\nAnswer:',
                'expected': 'yes',
                'alternatives': ['yes', 'Yes', 'YES'],
                'is_original': False,
            })
    
    # ============================================================
    # LEVEL 4: LONG-CONTEXT RECALL (100 prompts)
    # ============================================================
    level4_prompts = []
    
    # ============================================================
    # ORIGINAL PROMPTS (3) - From original test suite
    # ============================================================
    level4_prompts.append({
        'prompt': '''Question: What is Alice's favorite color?
Facts:
- Alice is 25 years old
- Alice lives in Paris
- Alice likes cats
- Alice's favorite color is blue
- Alice works as a teacher
- Alice speaks French

Question: What is Alice's favorite color?
Answer:''',
        'expected': 'blue',
        'alternatives': ['blue', 'Blue'],
        'is_original': True,
    })
    level4_prompts.append({
        'prompt': '''Question: What does Carol study?
Facts:
- Alice studies math
- Bob studies physics
- Carol studies chemistry
- David studies biology
- Emma studies history

Question: What does Carol study?
Answer:''',
        'expected': 'chemistry',
        'alternatives': ['chemistry', 'Chemistry'],
        'is_original': True,
    })
    level4_prompts.append({
        'prompt': '''Question: What is the 4th item?
List:
1. apple
2. banana
3. cherry
4. date
5. elderberry

Question: What is the 4th item in the list?
Answer:''',
        'expected': 'date',
        'alternatives': ['date', 'Date'],
        'is_original': True,
    })
    
    # ============================================================
    # NEW PROMPTS (97) - Generated for Mamba-friendly testing
    # ============================================================
    subjects = ['math', 'physics', 'chemistry', 'biology', 'history', 'geography', 'art', 'music', 'literature', 'science']
    cities_list = ['Paris', 'London', 'Tokyo', 'New York', 'Berlin', 'Madrid', 'Rome', 'Vienna', 'Moscow', 'Sydney']
    occupations = ['teacher', 'doctor', 'engineer', 'nurse', 'lawyer', 'artist', 'musician', 'writer', 'chef', 'pilot']
    fruits = ['apple', 'banana', 'cherry', 'date', 'elderberry', 'fig', 'grape', 'honeydew', 'kiwi', 'lemon']
    
    for i in range(97):
        pattern = i % 10
        if pattern == 0:  # Person attributes
            name = random.choice(names)
            age = random.choice(numbers[20:50])
            city = random.choice(cities_list)
            color = random.choice(colors)
            occupation = random.choice(occupations)
            animal = random.choice(['cats', 'dogs', 'birds'])
            language = random.choice(['French', 'Spanish', 'German', 'Italian', 'Japanese'])
            level4_prompts.append({
                'prompt': f'''Question: What is {name}'s favorite color?
Facts:
- {name} is {age} years old
- {name} lives in {city}
- {name} likes {animal}
- {name}'s favorite color is {color}
- {name} works as a {occupation}
- {name} speaks {language}

Question: What is {name}'s favorite color?
Answer:''',
                'expected': color,
                'alternatives': [color, color.capitalize()],
                'is_original': False,
            })
        elif pattern == 1:  # Study subjects
            name_list = random.sample(names, 5)
            subject_list = random.sample(subjects, 5)
            target_name = random.choice(name_list)
            target_idx = name_list.index(target_name)
            target_subject = subject_list[target_idx]
            facts = '\n'.join([f'- {name} studies {subj}' for name, subj in zip(name_list, subject_list)])
            level4_prompts.append({
                'prompt': f'''Question: What does {target_name} study?
Facts:
{facts}

Question: What does {target_name} study?
Answer:''',
                'expected': target_subject,
                'alternatives': [target_subject, target_subject.capitalize()],
                'is_original': False,
            })
        elif pattern == 2:  # List position
            item_list = random.sample(fruits, 5)
            position = random.choice([1, 2, 3, 4, 5])
            list_str = '\n'.join([f'{i+1}. {item}' for i, item in enumerate(item_list)])
            level4_prompts.append({
                'prompt': f'''Question: What is the {position}{"st" if position == 1 else "nd" if position == 2 else "rd" if position == 3 else "th"} item?
List:
{list_str}

Question: What is the {position}{"st" if position == 1 else "nd" if position == 2 else "rd" if position == 3 else "th"} item in the list?
Answer:''',
                'expected': item_list[position-1],
                'alternatives': [item_list[position-1], item_list[position-1].capitalize()],
                'is_original': False,
            })
        elif pattern == 3:  # Age recall
            name = random.choice(names)
            age = random.choice(numbers[20:50])
            city = random.choice(cities_list)
            color = random.choice(colors)
            occupation = random.choice(occupations)
            level4_prompts.append({
                'prompt': f'''Question: How old is {name}?
Facts:
- {name} is {age} years old
- {name} lives in {city}
- {name}'s favorite color is {color}
- {name} works as a {occupation}

Question: How old is {name}?
Answer:''',
                'expected': str(age),
                'alternatives': [str(age), f'{age} years old'],
                'is_original': False,
            })
        elif pattern == 4:  # City recall
            name = random.choice(names)
            age = random.choice(numbers[20:50])
            city = random.choice(cities_list)
            color = random.choice(colors)
            occupation = random.choice(occupations)
            level4_prompts.append({
                'prompt': f'''Question: Where does {name} live?
Facts:
- {name} is {age} years old
- {name} lives in {city}
- {name}'s favorite color is {color}
- {name} works as a {occupation}

Question: Where does {name} live?
Answer:''',
                'expected': city,
                'alternatives': [city, city.lower()],
                'is_original': False,
            })
        elif pattern == 5:  # Occupation recall
            name = random.choice(names)
            age = random.choice(numbers[20:50])
            city = random.choice(cities_list)
            color = random.choice(colors)
            occupation = random.choice(occupations)
            level4_prompts.append({
                'prompt': f'''Question: What is {name}'s job?
Facts:
- {name} is {age} years old
- {name} lives in {city}
- {name}'s favorite color is {color}
- {name} works as a {occupation}

Question: What is {name}'s job?
Answer:''',
                'expected': occupation,
                'alternatives': [occupation, occupation.capitalize()],
                'is_original': False,
            })
        elif pattern == 6:  # Number sequence
            num_list = random.sample(numbers[:20], 5)
            position = random.choice([1, 2, 3, 4, 5])
            list_str = '\n'.join([f'{i+1}. {num}' for i, num in enumerate(num_list)])
            level4_prompts.append({
                'prompt': f'''Question: What is the {position}{"st" if position == 1 else "nd" if position == 2 else "rd" if position == 3 else "th"} number?
List:
{list_str}

Question: What is the {position}{"st" if position == 1 else "nd" if position == 2 else "rd" if position == 3 else "th"} number?
Answer:''',
                'expected': str(num_list[position-1]),
                'alternatives': [str(num_list[position-1])],
                'is_original': False,
            })
        elif pattern == 7:  # Color list
            color_list = random.sample(colors, 5)
            position = random.choice([1, 2, 3, 4, 5])
            list_str = '\n'.join([f'{i+1}. {color}' for i, color in enumerate(color_list)])
            level4_prompts.append({
                'prompt': f'''Question: What is the {position}{"st" if position == 1 else "nd" if position == 2 else "rd" if position == 3 else "th"} color?
List:
{list_str}

Question: What is the {position}{"st" if position == 1 else "nd" if position == 2 else "rd" if position == 3 else "th"} color?
Answer:''',
                'expected': color_list[position-1],
                'alternatives': [color_list[position-1], color_list[position-1].capitalize()],
                'is_original': False,
            })
        elif pattern == 8:  # Animal preferences
            name = random.choice(names)
            age = random.choice(numbers[20:50])
            city = random.choice(cities_list)
            animal = random.choice(['cats', 'dogs', 'birds', 'fish', 'rabbits'])
            color = random.choice(colors)
            occupation = random.choice(occupations)
            level4_prompts.append({
                'prompt': f'''Question: What animals does {name} like?
Facts:
- {name} is {age} years old
- {name} lives in {city}
- {name} likes {animal}
- {name}'s favorite color is {color}
- {name} works as a {occupation}

Question: What animals does {name} like?
Answer:''',
                'expected': animal,
                'alternatives': [animal, animal.capitalize()],
                'is_original': False,
            })
        else:  # pattern == 9: Language recall
            name = random.choice(names)
            age = random.choice(numbers[20:50])
            city = random.choice(cities_list)
            language = random.choice(['French', 'Spanish', 'German', 'Italian', 'Japanese', 'Chinese', 'Korean'])
            color = random.choice(colors)
            occupation = random.choice(occupations)
            level4_prompts.append({
                'prompt': f'''Question: What language does {name} speak?
Facts:
- {name} is {age} years old
- {name} lives in {city}
- {name} speaks {language}
- {name}'s favorite color is {color}
- {name} works as a {occupation}

Question: What language does {name} speak?
Answer:''',
                'expected': language,
                'alternatives': [language, language.lower()],
                'is_original': False,
            })
    
    # ============================================================
    # LEVEL 5: COMBINED REASONING + MEMORY (100 prompts)
    # ============================================================
    level5_prompts = []
    
    # ============================================================
    # ORIGINAL PROMPTS (3) - From original test suite
    # ============================================================
    level5_prompts.append({
        'prompt': '''Question: How old is the oldest person?
Facts:
- Alice is 25 years old
- Bob is older than Alice
- Bob is 30 years old
- Carol is younger than Alice

Question: Who is the oldest person and how old are they?
Answer:''',
        'expected': 'Bob',
        'alternatives': ['Bob', 'bob', '30', 'Bob is 30', 'Bob, 30'],
        'is_original': True,
    })
    level5_prompts.append({
        'prompt': '''Question: What is the total cost?
Shopping list:
- Apples: $3 (bought by Alice)
- Bread: $2 (bought by Bob)
- Cheese: $5 (bought by Alice)

Question: How much did Alice spend in total?
Answer:''',
        'expected': '8',
        'alternatives': ['8', '$8', '8 dollars', 'eight'],
        'is_original': True,
    })
    level5_prompts.append({
        'prompt': '''Question: Can Alice reach the top shelf?
Facts:
- Top shelf is 6 feet high
- Alice is 5 feet tall
- Bob is 6.5 feet tall
- Alice can reach 1 foot above her height

Question: Can Alice reach the top shelf?
Answer:''',
        'expected': 'yes',
        'alternatives': ['yes', 'Yes', 'YES'],
        'is_original': True,
    })
    
    # ============================================================
    # NEW PROMPTS (97) - Generated for Mamba-friendly testing
    # ============================================================
    for i in range(97):
        pattern = i % 10
        if pattern == 0:  # Age comparison + recall
            name_list = random.sample(names, 3)
            ages = sorted(random.sample(numbers[20:50], 3), reverse=True)
            level5_prompts.append({
                'prompt': f'''Question: Who is oldest?
Facts:
- {name_list[0]} is {ages[0]} years old
- {name_list[1]} is {ages[1]} years old
- {name_list[2]} is {ages[2]} years old

Question: Who is the oldest person and how old are they?
Answer:''',
                'expected': name_list[0],
                'alternatives': [name_list[0], name_list[0].lower(), str(ages[0]), f'{name_list[0]} is {ages[0]}'],
                'is_original': False,
            })
        elif pattern == 1:  # Selective arithmetic
            name_list = random.sample(names, 3)
            prices = random.sample(numbers[:10], 3)
            items_list = ['apples', 'bread', 'cheese', 'milk', 'eggs']
            item_list = random.sample(items_list, 3)
            target_name = random.choice(name_list)
            target_items = [item for item, name in zip(item_list, name_list) if name == target_name]
            target_prices = [price for price, name in zip(prices, name_list) if name == target_name]
            total = sum(target_prices)
            shopping = '\n'.join([f'- {item.capitalize()}: ${price} (bought by {name})' for item, price, name in zip(item_list, prices, name_list)])
            level5_prompts.append({
                'prompt': f'''Question: What is the total cost?
Shopping list:
{shopping}

Question: How much did {target_name} spend in total?
Answer:''',
                'expected': str(total),
                'alternatives': [str(total), f'${total}', f'{total} dollars'],
                'is_original': False,
            })
        elif pattern == 2:  # Multi-step arithmetic reasoning
            name = random.choice(names)
            height = random.choice(numbers[4:7])
            shelf_height = random.choice(numbers[6:9])
            reach_above = random.choice([1, 1.5, 2])
            can_reach = (height + reach_above) >= shelf_height
            level5_prompts.append({
                'prompt': f'''Question: Can {name} reach the top shelf?
Facts:
- Top shelf is {shelf_height} feet high
- {name} is {height} feet tall
- {name} can reach {reach_above} foot{"s" if reach_above > 1 else ""} above their height

Question: Can {name} reach the top shelf?
Answer:''',
                'expected': 'yes' if can_reach else 'no',
                'alternatives': ['yes' if can_reach else 'no', 'Yes' if can_reach else 'No', 'YES' if can_reach else 'NO'],
                'is_original': False,
            })
        elif pattern == 3:  # Count items by person
            name_list = random.sample(names, 2)
            items_list = ['apples', 'bananas', 'oranges', 'grapes']
            item_counts = []
            for name in name_list:
                for item in random.sample(items_list, 2):
                    count = random.choice(numbers[1:5])
                    item_counts.append((item, count, name))
            target_name = random.choice(name_list)
            target_total = sum(count for item, count, name in item_counts if name == target_name)
            shopping = '\n'.join([f'- {item.capitalize()}: {count} (bought by {name})' for item, count, name in item_counts])
            level5_prompts.append({
                'prompt': f'''Question: How many items?
Shopping list:
{shopping}

Question: How many items did {target_name} buy in total?
Answer:''',
                'expected': str(target_total),
                'alternatives': [str(target_total)],
                'is_original': False,
            })
        elif pattern == 4:  # Price comparison
            name_list = random.sample(names, 3)
            prices = sorted(random.sample(numbers[10:50], 3), reverse=True)
            items_list = ['car', 'bike', 'phone', 'laptop', 'tablet']
            item_list = random.sample(items_list, 3)
            facts = '\n'.join([f'- {name} bought a {item} for ${price}' for name, item, price in zip(name_list, item_list, prices)])
            level5_prompts.append({
                'prompt': f'''Question: Who spent the most?
Facts:
{facts}

Question: Who spent the most money?
Answer:''',
                'expected': name_list[0],
                'alternatives': [name_list[0], name_list[0].lower()],
                'is_original': False,
            })
        elif pattern == 5:  # Age difference calculation
            name1, name2 = random.sample(names, 2)
            age1, age2 = random.sample(numbers[20:50], 2)
            if age1 < age2:
                age1, age2 = age2, age1
            diff = age1 - age2
            level5_prompts.append({
                'prompt': f'''Question: What is the age difference?
Facts:
- {name1} is {age1} years old
- {name2} is {age2} years old

Question: How many years older is {name1} than {name2}?
Answer:''',
                'expected': str(diff),
                'alternatives': [str(diff), f'{diff} years'],
                'is_original': False,
            })
        elif pattern == 6:  # Total cost across multiple people
            name_list = random.sample(names, 2)
            items_list = ['apples', 'bread', 'cheese']
            prices = random.sample(numbers[2:8], 3)
            shopping = '\n'.join([f'- {item.capitalize()}: ${price} (bought by {random.choice(name_list)})' for item, price in zip(items_list, prices)])
            total = sum(prices)
            level5_prompts.append({
                'prompt': f'''Question: What is the total cost?
Shopping list:
{shopping}

Question: How much was spent in total by everyone?
Answer:''',
                'expected': str(total),
                'alternatives': [str(total), f'${total}', f'{total} dollars'],
                'is_original': False,
            })
        elif pattern == 7:  # Height comparison with calculation
            name_list = random.sample(names, 3)
            heights = sorted(random.sample([4, 5, 6, 7], 3), reverse=True)
            facts = '\n'.join([f'- {name} is {height} feet tall' for name, height in zip(name_list, heights)])
            level5_prompts.append({
                'prompt': f'''Question: Who is tallest?
Facts:
{facts}

Question: Who is the tallest person?
Answer:''',
                'expected': name_list[0],
                'alternatives': [name_list[0], name_list[0].lower()],
                'is_original': False,
            })
        elif pattern == 8:  # Average calculation
            name_list = random.sample(names, 3)
            ages = random.sample(numbers[20:50], 3)
            avg = sum(ages) // 3
            facts = '\n'.join([f'- {name} is {age} years old' for name, age in zip(name_list, ages)])
            level5_prompts.append({
                'prompt': f'''Question: What is the average age?
Facts:
{facts}

Question: What is the average age of these three people?
Answer:''',
                'expected': str(avg),
                'alternatives': [str(avg), f'{avg} years old'],
                'is_original': False,
            })
        else:  # pattern == 9: Combined selection and arithmetic
            name_list = random.sample(names, 2)
            items_list = ['apples', 'bananas', 'oranges']
            prices = random.sample(numbers[2:6], 3)
            target_name = random.choice(name_list)
            target_items = random.sample(items_list, 2)
            shopping = '\n'.join([f'- {item.capitalize()}: ${price} (bought by {target_name if item in target_items else name_list[1]})' for item, price in zip(items_list, prices)])
            total = sum(price for item, price in zip(items_list, prices) if item in target_items)
            level5_prompts.append({
                'prompt': f'''Question: What is the total cost?
Shopping list:
{shopping}

Question: How much did {target_name} spend in total?
Answer:''',
                'expected': str(total),
                'alternatives': [str(total), f'${total}', f'{total} dollars'],
                'is_original': False,
            })
    
    # ============================================================
    # LEVEL 6: STRESS TEST (100 prompts)
    # ============================================================
    level6_prompts = []
    
    # ============================================================
    # ORIGINAL PROMPTS (2) - From original test suite
    # ============================================================
    level6_prompts.append({
        'prompt': '''Question: What is person E's occupation?
Database:
- Person A: Age 25, City Paris, Occupation Engineer
- Person B: Age 30, City London, Occupation Doctor
- Person C: Age 35, City Berlin, Occupation Teacher
- Person D: Age 28, City Madrid, Occupation Nurse
- Person E: Age 32, City Rome, Occupation Architect
- Person F: Age 27, City Vienna, Occupation Lawyer

Question: What is person E's occupation?
Answer:''',
        'expected': 'Architect',
        'alternatives': ['Architect', 'architect'],
        'is_original': True,
    })
    level6_prompts.append({
        'prompt': '''Question: Who has the blue car?
Garage inventory:
- Slot 1: Red car owned by Alice
- Slot 2: Blue car owned by Bob
- Slot 3: Green car owned by Carol
- Slot 4: Yellow car owned by David
- Slot 5: Black car owned by Emma

Question: Who owns the blue car?
Answer:''',
        'expected': 'Bob',
        'alternatives': ['Bob', 'bob'],
        'is_original': True,
    })
    
    # ============================================================
    # NEW PROMPTS (98) - Generated for Mamba-friendly testing
    # ============================================================
    person_labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']
    for i in range(98):
        pattern = i % 5
        if pattern == 0:  # Person database
            target_label = random.choice(person_labels[:6])
            target_idx = person_labels.index(target_label)
            name_list = random.sample(names, 6)
            age_list = random.sample(numbers[20:50], 6)
            city_list = random.sample(cities_list, 6)
            occupation_list = random.sample(occupations, 6)
            target_occupation = occupation_list[target_idx]
            database = '\n'.join([f'- Person {label}: Age {age}, City {city}, Occupation {occ}' 
                                 for label, age, city, occ in zip(person_labels[:6], age_list, city_list, occupation_list)])
            level6_prompts.append({
                'prompt': f'''Question: What is person {target_label}'s occupation?
Database:
{database}

Question: What is person {target_label}'s occupation?
Answer:''',
                'expected': target_occupation,
                'alternatives': [target_occupation, target_occupation.capitalize()],
                'is_original': False,
            })
        elif pattern == 1:  # Garage inventory
            target_color = random.choice(colors)
            name_list = random.sample(names, 5)
            color_list = random.sample(colors, 5)
            if target_color not in color_list:
                color_list[0] = target_color
            target_name = name_list[color_list.index(target_color)]
            inventory = '\n'.join([f'- Slot {i+1}: {color.capitalize()} car owned by {name}' 
                                  for i, (color, name) in enumerate(zip(color_list, name_list))])
            level6_prompts.append({
                'prompt': f'''Question: Who has the {target_color} car?
Garage inventory:
{inventory}

Question: Who owns the {target_color} car?
Answer:''',
                'expected': target_name,
                'alternatives': [target_name, target_name.lower()],
                'is_original': False,
            })
        elif pattern == 2:  # Student records
            name_list = random.sample(names, 6)
            subject_list = random.sample(subjects, 6)
            # Use choices to allow duplicates since we need 6 grades from 5 options
            grade_list = random.choices(['A', 'B', 'C', 'D', 'F'], k=6)
            target_name = random.choice(name_list)
            target_idx = name_list.index(target_name)
            target_grade = grade_list[target_idx]
            records = '\n'.join([f'- {name}: Subject {subj}, Grade {grade}' 
                                for name, subj, grade in zip(name_list, subject_list, grade_list)])
            level6_prompts.append({
                'prompt': f'''Question: What is {target_name}'s grade?
Student records:
{records}

Question: What is {target_name}'s grade?
Answer:''',
                'expected': target_grade,
                'alternatives': [target_grade],
                'is_original': False,
            })
        elif pattern == 3:  # Product catalog
            item_list = random.sample(items, 6)
            price_list = random.sample(numbers[10:100], 6)
            color_list = random.sample(colors, 6)
            target_item = random.choice(item_list)
            target_idx = item_list.index(target_item)
            target_price = price_list[target_idx]
            catalog = '\n'.join([f'- {item.capitalize()}: ${price}, Color {color}' 
                               for item, price, color in zip(item_list, price_list, color_list)])
            level6_prompts.append({
                'prompt': f'''Question: What is the price of {target_item}?
Product catalog:
{catalog}

Question: What is the price of {target_item}?
Answer:''',
                'expected': str(target_price),
                'alternatives': [str(target_price), f'${target_price}'],
                'is_original': False,
            })
        else:  # pattern == 4: Team roster
            name_list = random.sample(names, 6)
            # Use choices to allow duplicates since we need 6 positions from 4 options
            position_list = random.choices(['forward', 'midfielder', 'defender', 'goalkeeper'], k=6)
            number_list = random.sample(numbers[1:99], 6)
            target_name = random.choice(name_list)
            target_idx = name_list.index(target_name)
            target_number = number_list[target_idx]
            roster = '\n'.join([f'- {name}: Position {pos}, Number {num}' 
                              for name, pos, num in zip(name_list, position_list, number_list)])
            level6_prompts.append({
                'prompt': f'''Question: What is {target_name}'s number?
Team roster:
{roster}

Question: What is {target_name}'s jersey number?
Answer:''',
                'expected': str(target_number),
                'alternatives': [str(target_number)],
                'is_original': False,
            })
    
    return {
        'level1_simple_recall': level1_prompts,
        'level2_two_hop': level2_prompts,
        'level3_three_hop': level3_prompts,
        'level4_long_context': level4_prompts,
        'level5_combined': level5_prompts,
        'level6_stress_test': level6_prompts,
    }

