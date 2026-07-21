#!/usr/bin/env python3
"""
Phase 2 — generate the labeled training dataset.

"Hand labeled" here means: every template's tier assignment is a genuine
judgment call made against the phase spec's own definitions (Tier 1 =
reformatting/extraction/basic Q&A, Tier 2 = summarization/classification/
structured analysis, Tier 3 = multi-step reasoning/creative/nuanced
judgement). Purely mechanical categories (capitals, arithmetic,
translation, extraction) are generated from small parallel lists to reach
volume without me hand-typing 70 near-duplicate sentences; everything in
Tier 2 and Tier 3 — where correctness of the label actually depends on
nuance — is written out explicitly, one prompt at a time.

Usage:
    uv run python scripts/generate_training_data.py
    uv run python scripts/generate_training_data.py --output data/classifier/training_data.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

# ── Tier 1 (simple): mechanical, templated ─────────────────────────────────────

_CAPITALS = {
    "France": "Paris",
    "Japan": "Tokyo",
    "Brazil": "Brasilia",
    "Canada": "Ottawa",
    "Egypt": "Cairo",
    "Australia": "Canberra",
    "Germany": "Berlin",
    "Kenya": "Nairobi",
    "Peru": "Lima",
    "Norway": "Oslo",
    "Italy": "Rome",
    "Spain": "Madrid",
    "Thailand": "Bangkok",
    "Mexico": "Mexico City",
    "Poland": "Warsaw",
}
_ADDITION_PAIRS = [(12, 7), (45, 23), (101, 99), (8, 16), (250, 75), (33, 67), (19, 42), (500, 125)]
_MULTIPLICATION_PAIRS = [(6, 7), (12, 11), (15, 4), (9, 9), (23, 3), (17, 5), (8, 12), (14, 6)]
_TRANSLATIONS = [
    ("Good morning, how are you?", "French"),
    ("Where is the train station?", "Spanish"),
    ("Thank you very much", "German"),
    ("See you tomorrow", "Italian"),
    ("I would like a coffee, please", "Japanese"),
    ("Happy birthday", "Portuguese"),
    ("Nice to meet you", "Mandarin Chinese"),
    ("What time is it?", "Korean"),
    ("I am learning to cook", "Russian"),
    ("The weather is nice today", "Dutch"),
]
_NAME_EMAIL_PAIRS = [
    ("Sarah Connor", "sarah.connor@example.com"),
    ("John Smith", "john.smith@company.org"),
    ("Maria Garcia", "m.garcia@studio.net"),
    ("Wei Zhang", "wei.zhang@research.edu"),
    ("Amara Okafor", "amara.okafor@ngo.org"),
    ("Liam O'Brien", "liam.obrien@finance.com"),
    ("Priya Sharma", "priya.sharma@tech.io"),
    ("Diego Fernandez", "diego.f@travel.com"),
]
_NAME_PHONE_PAIRS = [
    ("Anna Kowalski", "555-201-3344"),
    ("Tom Baker", "555-822-1190"),
    ("Fatima Al-Sayed", "555-467-2201"),
    ("Carlos Mendes", "555-903-6612"),
    ("Grace Lin", "555-334-7789"),
    ("Noah Bennett", "555-118-4402"),
]

_SIMPLE_MISC = [
    "Reformat this list into bullet points: apples, bananas, oranges, grapes",
    "Convert this sentence to uppercase: 'the meeting starts at nine'",
    "Convert '03/15/2024' to ISO 8601 date format.",
    "Rewrite this sentence in passive voice: 'The chef cooked the meal.'",
    "Copy the following text exactly: 'All systems operational.'",
    "Reformat this name from 'Smith, John' to 'John Smith'.",
    "Convert 5 kilometers to miles.",
    "Change this list of numbers to a comma-separated string: 4, 8, 15, 16, 23",
    "Reword this sentence to start with 'Because': 'The store closed early since it was a holiday.'",
    "Convert this temperature from Celsius to Fahrenheit: 20 degrees.",
    "What is the chemical symbol for gold?",
    "How many days are in February during a leap year?",
    "What is the plural of 'cactus'?",
    "Give me the dictionary definition of 'ubiquitous'.",
    "What year did the Berlin Wall fall?",
    "List the first five prime numbers.",
    "What is the boiling point of water in Celsius?",
    "Who wrote the play 'Romeo and Juliet'?",
    "What does the abbreviation 'CPU' stand for?",
    "How many continents are there?",
]

# ── Tier 2 (moderate): hand-written ────────────────────────────────────────────

_MODERATE = [
    "Summarize this paragraph in one sentence: The city council voted to approve a new public transit line that will connect the downtown core to the northern suburbs, with construction expected to begin next spring and take approximately eighteen months to complete.",
    "Summarize the following text in 3 bullet points: Remote work has changed how companies think about office space. Many organizations have reduced their real estate footprint, while others have redesigned offices to focus on collaboration rather than individual desks. Employee surveys show mixed opinions.",
    "Give a two-sentence summary of this article excerpt: The new smartphone features a larger battery, a faster processor, and an improved camera system. Reviewers have praised the display quality but noted the price increase may deter budget-conscious buyers.",
    "Summarize the main point of this email in one line: 'Hi team, following our discussion yesterday, we've decided to postpone the product launch by two weeks to allow more time for QA testing.'",
    "Condense this meeting transcript into key takeaways: 'We discussed the Q3 budget overrun, agreed to cut marketing spend by 10%, and decided to revisit hiring plans next quarter.'",
    "Summarize this product review in one sentence: 'I've been using this blender for three months now. It's powerful and easy to clean, though a bit loud. The price was higher than expected but the build quality justifies it.'",
    "Provide a brief summary of this news snippet: A wildfire in the northern region has been contained after three days, with no reported injuries, thanks to early evacuation orders and favorable wind conditions.",
    "Summarize this research abstract in plain language: The study examined the effects of intermittent fasting on metabolic markers in 200 participants over 12 weeks, finding modest improvements in insulin sensitivity but no significant weight change.",
    "Summarize this customer complaint in one sentence: 'I ordered a laptop two weeks ago and it still hasn't shipped. I've contacted support three times and gotten different answers each time.'",
    "Give a short summary of this policy update: Starting next month, all employees will be required to complete an annual cybersecurity training module, with completion tracked through the HR portal.",
    "Summarize this contract clause in simple terms: 'Either party may terminate this agreement with 30 days written notice, provided all outstanding invoices are settled prior to the termination date.'",
    "Summarize the plot of this short story premise in one sentence: A lighthouse keeper on a remote island discovers a message in a bottle that leads her to uncover a decades-old mystery.",
    "Summarize this weather report: Temperatures will drop significantly overnight with a chance of frost in low-lying areas, clearing by midday tomorrow with light winds.",
    "Summarize the key findings of this survey in two sentences: Of 500 respondents, 68% preferred hybrid work arrangements, 22% preferred fully remote, and 10% preferred fully in-office.",
    "Summarize this legal notice in plain English: 'Failure to respond within 14 days of receipt of this notice may result in default judgment being entered against you.'",
    "Summarize this technical changelog: Version 2.3 introduces dark mode, fixes a memory leak in the export function, and improves load times by roughly 40% on large datasets.",
    "Summarize the outcome of this sports match in one sentence: The home team trailed by ten points at halftime but rallied in the fourth quarter to win by three.",
    "Summarize this restaurant review: 'The service was attentive without being intrusive, the pasta was cooked perfectly, but the dessert menu was disappointingly limited.'",
    "Summarize this project status update: The frontend is 90% complete, backend integration is on track for next week, but QA may slip a few days due to two newly discovered bugs.",
    "Summarize this travel itinerary in a few words: Day 1 arrival and city tour, Day 2 museum visits and local market, Day 3 day trip to the coast, Day 4 departure.",
    "Classify the sentiment of this review as positive, negative, or neutral: 'The food was okay but the service was painfully slow.'",
    "Classify this email as spam or not spam: 'Congratulations! You've won a free cruise. Click here to claim your prize now!'",
    "Classify this customer feedback as a bug report, feature request, or general question: 'Could you add the ability to export reports as CSV files?'",
    "Classify the tone of this message as formal or informal: 'hey just checking if you got my last email, lmk when free'",
    "Classify this news headline by topic (politics, sports, technology, or entertainment): 'Local team clinches championship after dramatic overtime win.'",
    "Classify this product review as positive or negative: 'Absolutely love this. Best purchase I've made all year.'",
    "Classify this support ticket by urgency (low, medium, high): 'The entire payment system is down and customers can't check out.'",
    "Classify this social media comment as constructive criticism or a personal attack: 'Your argument ignores the counterexamples raised earlier in the thread.'",
    "Classify this job application excerpt as a strong fit or weak fit: 'I have five years of experience in the exact role you're hiring for and led a team of similar size.'",
    "Classify this text message as urgent or non-urgent: 'No rush, whenever you get a chance can you send me that file?'",
    "Classify this review's sentiment: 'The hotel room was clean but smaller than the photos suggested, and the wifi kept disconnecting.'",
    "Classify this article as an opinion piece or a factual news report.",
    "Categorize these items into fruits and vegetables: apple, carrot, banana, spinach, orange, broccoli.",
    "Classify this survey response as satisfied, neutral, or dissatisfied: 'It does what it's supposed to, nothing more, nothing less.'",
    "Classify this email thread as resolved or unresolved based on the last message: 'Thanks, that fixed it!'",
    "Classify this comment as a question, a complaint, or a compliment: 'This tutorial was exactly what I needed, thank you!'",
    "Categorize this list of expenses into categories like food, transport, and entertainment: taxi ride, movie ticket, groceries, bus pass, restaurant dinner.",
    "Classify this book review's overall sentiment: 'The pacing dragged in the middle but the ending redeemed the whole book.'",
    "Classify this warranty claim as valid or likely fraudulent based on the description given.",
    "Classify these customer support messages by department: billing, technical, or shipping.",
    "Compare these two cloud providers in a table with columns for pricing, uptime SLA, and support quality: AWS and Azure.",
    "Organize this data into categories by department: list which of these tasks belong to Engineering, Sales, Marketing, or HR — hiring, code review, ad campaigns, quarterly report, onboarding, deployment.",
    "Structure this list of ingredients into a shopping list organized by grocery aisle: milk, chicken breast, spinach, olive oil, cereal, apples, cheddar cheese.",
    "Compare these two smartphones based on battery life, camera quality, and price: Model A and Model B.",
    "Outline the steps of the onboarding process based on this description: New hires complete paperwork, then IT sets up equipment, then they meet their manager, then they attend orientation.",
    "Organize this feedback into pros and cons columns: 'The UI is intuitive but the app crashes occasionally. Load times are fast, but the search feature is unreliable.'",
    "Compare the nutritional profiles of an apple and a banana in a short table.",
    "Structure this meeting agenda into timed sections: budget review, hiring update, product roadmap, open discussion.",
    "Organize these tasks by priority (high, medium, low): fix production bug, update documentation, respond to customer email, plan next sprint.",
    "Compare renting versus buying a home across cost, flexibility, and long-term value.",
    "Categorize these expenses as fixed or variable costs: rent, electricity bill, groceries, insurance premium, dining out.",
    "Outline a basic project plan based on this goal: launch a new company website within two months.",
    "Compare two programming languages, Python and JavaScript, in terms of typical use cases.",
    "Organize this list of symptoms by likely severity: mild headache, chest pain, minor bruise, difficulty breathing.",
    "Structure this raw data into a table with columns for name, role, and start date: 'Jane, engineer, started March; Tom, designer, started June; Priya, manager, started January.'",
    "Summarize the key risks mentioned in this project brief: tight timeline, limited budget, and dependency on a third-party API that has had reliability issues.",
    "Classify this GitHub issue as a bug, enhancement, or documentation request: 'The README is missing setup instructions for Windows.'",
    "Compare in-house development versus outsourcing for a small startup building its first product.",
    "Summarize this quarterly earnings snippet in one sentence: Revenue grew 12% year over year while operating costs rose only 4%, resulting in improved margins.",
    "Classify this online comment as toxic or non-toxic: 'I disagree with your take but I see where you're coming from.'",
    "Organize this list of chores by frequency (daily, weekly, monthly): vacuuming, watering plants, paying bills, doing laundry, cleaning the oven.",
    "Summarize the results of this A/B test: Variant B increased click-through rate by 8% but decreased average session duration slightly.",
    "Classify this resume bullet point as a quantified achievement or a vague description: 'Responsible for improving team performance.'",
    "Compare two marketing channels, email and social media, for a small local business.",
    "Summarize this incident report in two sentences: A server outage occurred at 2am due to a failed disk, detected by monitoring within 5 minutes, with service restored by 3:30am via failover.",
    "Classify this app store review as a feature complaint or a bug complaint: 'Every time I try to upload a photo the app freezes.'",
    "Organize this list of skills into technical and soft skills: communication, Python, teamwork, SQL, leadership, data visualization.",
    "Summarize this customer interview note in one sentence: The user said onboarding took too long and they almost gave up before finding value in the product.",
]

# ── Tier 3 (complex): hand-written ─────────────────────────────────────────────

_COMPLEX = [
    "Analyze the pros and cons of remote work versus in-office work for a growing startup, and justify a recommendation.",
    "Given these constraints — a $50,000 budget, a 3-month deadline, and a team of four engineers — design a rough plan for building a minimum viable product for a scheduling app.",
    "Critique this business plan and suggest three specific improvements: 'We will sell handmade candles online, targeting all consumers, with no clear marketing budget and a plan to be profitable within one month.'",
    "Explain step by step how you would diagnose why a web application's response times suddenly doubled after a routine deployment.",
    "Evaluate the trade-offs between building a feature in-house versus buying a third-party solution, considering cost, control, and time-to-market.",
    "A retail company's sales dropped 15% in one quarter despite increased marketing spend. Analyze possible explanations and how you would investigate further.",
    "Assess the risks and benefits of a company switching its entire infrastructure from on-premise servers to the cloud.",
    "Explain the reasoning behind choosing a relational database over a NoSQL database for a financial transactions system.",
    "Analyze how rising interest rates might affect a small business that relies heavily on credit for inventory financing.",
    "Given this ambiguous bug report — 'the app is slow sometimes' — walk through how you would narrow down the root cause.",
    "Compare and critically evaluate two hiring strategies: promoting from within versus hiring externally for a leadership role.",
    "Design a high-level system architecture for a ride-sharing app, explaining the reasoning behind each major component.",
    "Analyze the ethical considerations of using customer data to train a recommendation algorithm without explicit opt-in consent.",
    "Explain why a startup might choose to delay fundraising even when investor interest is high, weighing the trade-offs involved.",
    "Diagnose the likely cause of a sudden spike in customer churn and outline what data you would need to confirm your hypothesis.",
    "Evaluate whether a company should prioritize fixing technical debt or shipping new features next quarter, and justify your reasoning.",
    "Analyze the long-term implications of a company adopting a four-day work week, considering productivity, morale, and client expectations.",
    "Given a scenario where two team members disagree on technical direction, explain how you would facilitate a resolution and what factors would guide the final decision.",
    "Critically assess the argument that automation will lead to net job losses over the next decade, presenting both supporting and opposing evidence.",
    "Design an experiment to test whether a new onboarding flow improves user retention, including what metrics you would track.",
    "Analyze why a previously successful product might be losing market share to a newer competitor, considering both product and market factors.",
    "Explain the trade-offs between optimizing a system for read-heavy versus write-heavy workloads, with an example of when each choice makes sense.",
    "Evaluate the pros and cons of a company going public versus staying privately funded.",
    "Given conflicting stakeholder priorities — engineering wants to reduce technical debt, sales wants new features — propose a prioritization framework and justify it.",
    "Analyze the potential unintended consequences of a city implementing a congestion pricing policy for its downtown core.",
    "Write a one-paragraph story about a robot who discovers music for the first time.",
    "Write a haiku about autumn leaves.",
    "Compose a short poem about the feeling of starting a new job.",
    "Write a short dialogue between two old friends meeting unexpectedly after ten years apart.",
    "Imagine a world where gravity briefly reverses every night, and write a short scene depicting how one family adapts.",
    "Write a brief fictional diary entry from the perspective of a lighthouse keeper during a storm.",
    "Brainstorm five creative names for a new coffee shop that specializes in single-origin pour-overs.",
    "Write a short story opening that immediately establishes tension between two characters without stating what the conflict is directly.",
    "Compose a limerick about a cat who believes it runs the household.",
    "Write a short piece of flash fiction, no more than 150 words, about a character finding an old photograph.",
    "Invent a myth explaining why the moon changes shape throughout the month.",
    "Write a monologue for a character who has just decided to leave their hometown for good.",
    "Compose a short poem from the perspective of the last leaf on a tree in winter.",
    "Write a scene where a character discovers a hidden room in a house they've lived in for years.",
    "Brainstorm a creative concept for a children's book about overcoming fear of the dark.",
    "Write a short story about a chef who can taste people's memories in the food they cook.",
    "Compose a piece of micro-fiction that ends with an unexpected twist, in under 100 words.",
    "Write a letter from a character to their future self, ten years from now.",
    "Imagine a city built entirely underwater and describe a day in the life of one of its residents.",
    "Write a short conversation between an inventor and their creation, a robot experiencing doubt for the first time.",
    "Critique this essay's argument and suggest how the author could strengthen their thesis: 'Social media is entirely responsible for declining attention spans in teenagers.'",
    "Recommend whether a friend should accept a lower-paying job with better work-life balance or stay in a high-paying but stressful role, and justify your reasoning.",
    "Discuss the ethical dilemma of an autonomous vehicle having to choose between two harmful outcomes in an unavoidable accident scenario.",
    "Evaluate whether a company's decision to lay off 10% of staff while announcing record profits the same quarter is ethically justifiable, considering multiple stakeholder perspectives.",
    "Give nuanced advice to someone deciding whether to confront a close friend about a pattern of unreliable behavior.",
    "Critique this argument for its logical weaknesses: 'Since most successful entrepreneurs dropped out of college, dropping out of college increases your chances of success.'",
    "Discuss both sides of the debate on whether standardized testing is a fair measure of student ability.",
    "Recommend how a manager should handle a high-performing employee who consistently disregards team processes, weighing the trade-offs of each approach.",
    "Evaluate the fairness of a company promoting based on tenure versus based on measurable performance.",
    "Discuss the tension between individual privacy and public safety in the context of city-wide surveillance cameras.",
    "Critique this policy proposal and identify its unstated assumptions: 'We should ban all remote work to increase collaboration.'",
    "Give balanced advice on whether someone should prioritize paying off debt aggressively or investing for the future, given uncertain income stability.",
    "Discuss the ethical considerations of a company using AI to screen job applicants without human review.",
    "Evaluate whether it's ethical for a journalist to use a hidden recording to expose wrongdoing, weighing the public interest against consent.",
    "Recommend how a small nonprofit should allocate a limited grant between direct services and long-term capacity building, and explain the trade-offs.",
    "Critique the reasoning in this statement and suggest a more balanced view: 'Failure is always the best teacher, so mistakes should never be avoided.'",
    "Discuss whether a company should disclose an internal security breach immediately or wait until an investigation is complete, weighing transparency against accuracy.",
    "Give a nuanced take on whether artificial intelligence should be allowed to make final decisions in medical diagnoses without physician review.",
    "Evaluate the argument that remote-first companies will outcompete in-office companies for talent over the next decade.",
    "Recommend whether a startup founder should take on a co-founder at this stage, given the trade-offs of equity dilution versus shared workload.",
    "A train leaves station A at 60 mph heading toward station B, 300 miles away. Another train leaves station B at the same time heading toward station A at 40 mph. Explain step by step when and where they will meet.",
    "A company's revenue grows 20% in year one and declines 10% in year two from a starting point of $500,000. Walk through the calculation to find the revenue at the end of year two, explaining each step.",
    "You have three boxes of different weights and a balance scale with only two weighings allowed. Explain a strategy to determine which box is the heaviest.",
    "A recipe serves 4 people and requires 2.5 cups of flour. Walk through how to adjust the recipe to serve 15 people, explaining your reasoning.",
    "Explain step by step how to solve for x in the equation 3x + 7 = 2x - 5, showing your reasoning at each stage.",
    "A store offers a 20% discount, then an additional 10% off the discounted price. Explain step by step whether this is the same as a flat 30% discount, and why or why not.",
    "Three friends split a restaurant bill of $126 unevenly based on what they ordered. Explain a fair method to calculate each person's share.",
    "Walk through the reasoning for whether it's cheaper to buy a $1,200 laptop outright or finance it over 12 months at 8% annual interest.",
    "Explain step by step how you would estimate the number of piano tuners in a large city, showing your reasoning at each stage.",
    "A tank is filled by pipe A in 4 hours and drained by pipe B in 6 hours. If both pipes are open at the same time, explain step by step how long it takes to fill the tank.",
]


def _build_simple() -> list[str]:
    prompts = [f"What is the capital of {country}?" for country in _CAPITALS]
    prompts += [f"What is {a} + {b}?" for a, b in _ADDITION_PAIRS]
    prompts += [f"What is {a} multiplied by {b}?" for a, b in _MULTIPLICATION_PAIRS]
    prompts += [f"Translate '{phrase}' into {language}." for phrase, language in _TRANSLATIONS]
    prompts += [
        f"Extract the email address from this text: 'Hi, this is {name}, you can reach me at "
        f"{email} for any questions.'"
        for name, email in _NAME_EMAIL_PAIRS
    ]
    prompts += [
        f"Extract the person's name and phone number from this text: '{name} can be reached at "
        f"{phone} regarding the delivery.'"
        for name, phone in _NAME_PHONE_PAIRS
    ]
    prompts += _SIMPLE_MISC
    return prompts


def build_dataset() -> list[dict[str, str]]:
    examples: list[dict[str, str]] = []
    for prompt in _build_simple():
        examples.append({"prompt": prompt, "tier": "simple"})
    for prompt in _MODERATE:
        examples.append({"prompt": prompt, "tier": "moderate"})
    for prompt in _COMPLEX:
        examples.append({"prompt": prompt, "tier": "complex"})
    return examples


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="data/classifier/training_data.jsonl")
    parser.add_argument("--seed", type=int, default=42, help="Shuffle seed for reproducibility")
    args = parser.parse_args()

    examples = build_dataset()
    random.Random(args.seed).shuffle(examples)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        for example in examples:
            f.write(json.dumps(example) + "\n")

    counts: dict[str, int] = {}
    for example in examples:
        counts[example["tier"]] = counts.get(example["tier"], 0) + 1

    print(f"Wrote {len(examples)} examples to {output_path}")
    for tier, count in sorted(counts.items()):
        print(f"  {tier:<10} {count}")


if __name__ == "__main__":
    main()
