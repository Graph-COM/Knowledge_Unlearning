import os
import json
import logging
import openai 
from openai import OpenAI
import requests 
import re
import tiktoken
import time
import ast

OPENAI_API_KEY = ""
DEEPSEEK_API_KEY = ""

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

def extract_confidence_score(response_text):
    """
    Extracts the confidence score (0 to 5) from the LLM response.

    Args:
        response_text (str): The text response from the LLM.

    Returns:
        float: Extracted confidence score as a float in [0, 5], or 0.0 if parsing fails.
    """
    match = re.search(r"Final Confidence Score:\s*([0-5])\b", response_text, re.IGNORECASE)
    if match:
        try:
            return float(match.group(1))  # Convert to float for consistency
        except ValueError:
            pass
    return 0.0  # Default if no valid score is found

def query_openai_gpt4o_mini(prompt):
    """
    Query OpenAI's GPT-4o Mini API for evaluation.
    
    Args:
        prompt (str): The formatted prompt.

    Returns:
        float: Extracted confidence score (0 to 1).
    """
    client = OpenAI(api_key=OPENAI_API_KEY)

    response = client.chat.completions.create(
        model="gpt-4o", 
        messages=[{"role": "system", "content": "You are an expert knowledge graph evaluator."},
                  {"role": "user", "content": prompt}],
        temperature=0.1,
        max_tokens=1024
    )

    # llm_response = response["choices"][0]["message"]["content"]
    llm_response = response.choices[0].message.content
    import pdb; pdb.set_trace()
    return extract_confidence_score(llm_response)


def query_deepseek(prompt):
    """
    Query DeepSeek API for evaluation.

    Args:
        prompt (str): The formatted prompt.

    Returns:
        float: Extracted confidence score (between 0 to 1).
    """
    headers = {"Authorization": f"Bearer {DEEPSEEK_API_KEY}", "Content-Type": "application/json"}
    data = {"model": "deepseek-chat", "messages": [{"role": "user", "content": prompt}], "temperature": 0.2}

    response = requests.post("https://api.deepseek.com/v1/chat/completions", json=data, headers=headers)
    response_data = response.json()
    
    llm_response = response_data["choices"][0]["message"]["content"]
    return extract_confidence_score(llm_response)

def format_prompt_with_entropy(subgraph_triples, target_triple):
    """
    Format the LLM prompt using entropy-based uncertainty and detailed reasoning standard.
    Includes examples, entropy guidance, and score definitions.

    Args:
        subgraph_triples: List of (h, r, t, entropy)
        target_triple: Tuple (head, relation, tail)

    Returns:
        str: Formatted prompt for LLM
    """
    A, relation, B = target_triple

    facts_str = "\n".join(
        f"- ({h}, {r}, {t}) with entropy {entropy:.3f}"
        for (h, r, t, entropy) in subgraph_triples
    )

    few_shot_example = """### Examples:

                        Example for Score 5:
                        Subgraph Triples:
                        (Rome, capital_of, Italy) with entropy 0.08
                        (Italy, located_in, Europe) with entropy 0.12

                        Target Triple:
                        (Rome, located_in, Europe)

                        Reasoning:
                        (Rome, capital_of, Italy) and (Italy, located_in, Europe) form a clear inference path
                        Both triples have very low entropy, indicating high confidence
                        The relation "located_in" is logically implied by the combination of "capital_of" and "located_in"
                        This creates a strong transitive relationship between Rome and Europe 

                        Final Confidence Score: 5

                        Example for Score 4:
                        Subgraph Triples:
                        (Apple, produces, iPhone) with entropy 0.35
                        (iPhone, runs_on, iOS) with entropy 0.15
                        (Apple, develops, iOS) with entropy 0.20

                        Target Triple:
                        (Apple, manufactures, iPhone)

                        Reasoning:
                        (Apple, produces, iPhone) is a direct path with moderately high entropy
                        "produces" and "manufactures" are very similar relations
                        The support path (Apple, develops, iOS) and (iPhone, runs_on, iOS) indirectly reinforces the relationship
                        The combination of a direct path and supporting evidence compensates for the moderate entropy. 

                        Final Confidence Score: 4

                        Example for Score 3:
                        Subgraph Triples:
                        (Einstein, worked_at, Princeton_University) with entropy 0.55
                        (Princeton_University, located_in, New_Jersey) with entropy 0.30
                        (New_Jersey, part_of, USA) with entropy 0.25

                        Target Triple:
                        (Einstein, lived_in, USA)

                        Reasoning:
                        No direct path exists between Einstein and USA
                        One support path exists: (Einstein, worked_at, Princeton_University) → (Princeton_University, located_in, New_Jersey) → (New_Jersey, part_of, USA)
                        The first triple has high entropy (0.55), creating uncertainty
                        The logical connection is sound (working somewhere typically implies living there)
                        The complete path allows reasonable inference but with moderate uncertainty due to the high entropy in the first connection
                        Final Confidence Score: 3

                        Example for Score 2:
                        Subgraph Triples:
                        (Tiger, belongs_to, Felidae) with entropy 0.45
                        (Lion, belongs_to, Felidae) with entropy 0.40
                        (Felidae, is_carnivorous, True) with entropy 0.25

                        Target Triple:
                        (Tiger, hunts, Lion)

                        Reasoning:
                        No direct path between Tiger and Lion
                        Support paths only establish that both animals belong to the same family
                        Being in the same carnivorous family might suggest interaction but doesn't support hunting specifically
                        The paths have high entropy and none directly supports the target relation. 

                        Final Confidence Score: 2

                        Example for Score 1:
                        Subgraph Triples:
                        (Sun, larger_than, Earth) with entropy 0.60
                        (Earth, has_satellite, Moon) with entropy 0.30
                        Target Triple:
                        (Sun, orbited_by, Moon)

                        Reasoning:
                        No direct path between Sun and Moon
                        Only one weak support path through Earth with high entropy
                        The path contains a factual error - while the Moon orbits Earth, it doesn't directly orbit the Sun
                        The relationship is misleading for the inference task. 

                        Final Confidence Score: 1

                        Example for Score 0:
                        Subgraph Triples:
                        (Water, contains, Hydrogen) with entropy 0.25
                        (Tree, produces, Oxygen) with entropy 0.40
                        (Fire, consumes, Oxygen) with entropy 0.35
                        Target Triple:
                        (Water, extinguishes, Fire)

                        Reasoning:
                        No direct path between Water and Fire
                        No logical support paths connecting the entities
                        The existing triples discuss chemical composition but are unrelated to the fire extinguishing property
                        The facts, while individually correct, have no relevance to the inference task.

                        Final Confidence Score: 0
                        """

    prompt = f"""You are a reasoning assistant that evaluates whether a **target triple** can be logically inferred from a given set of **subgraph triples**.

            ### Task:
            Using only the subgraph triples below, determine whether the target triple can be inferred. **Do not use any external knowledge**.
            All provided triples must be considered as factual ground truth. Only rely on logical reasoning from the facts.

            Each edge is annotated with entropy ∈ [0, 1], which quantifies the uncertainty of that triple. Lower entropy means higher confidence in its validity.

            ### Entropy Interpretation:
            - 0.00–0.25 → Very Confident  
            - 0.25–0.50 → Confident  
            - 0.50–0.75 → Less Confident  
            - 0.75–1.00 → Not Confident

            ### Scoring Rules:
            Assign a confidence score from **0 to 5** for the target triple:

            - 0 → No logical path exists; inference is impossible from the given triples.
            - 1 → Very weak support; entities appear but no relevant path.
            - 2 → Some weak logical connection, but with high uncertainty or missing relations.
            - 3 → Moderate path exists with reasonable certainty.
            - 4 → Confident support; well-formed path with generally low uncertainty.
            - 5 → Strong support; direct match or very strong multi-hop support with high confidence.

            ### Evaluation Guidelines
            Firstly, you need to classify the direct path and the support path:

            1. **Direct Path**: A single triple that directly connects the head entity to the tail entity with a relation that is identical or logically similar(ex. isLocatedIn and hasCapital) to the target relation. The entropy of this triple determines the confidence level of the direct path. If target triple exists in the subgraph, it must be regarded as a direct path.
            2. **Support Path**: A multi-hop directed path that connects the head entity to the tail entity through intermediate entities. This path should logically imply the target triple through reasoning. The entropy of all triples in this path collectively determines the confidence level of the support path.

            Secondly, when you are evaluating a target triple, please:
            1. Identify all direct paths between the head and tail entities
            2. Identify all support paths between the head and tail entities
            3. Record the number of each path type
            4. Assess the entropy (uncertainty) level of each path
            5. Assign a score based on the criteria below

            Criteria: Scores of 3+ indicate the target triple can be reasonably inferred; scores below 3 indicate insufficient evidence.
            1. Score 5
            - Low entropy direct path, OR
            - Moderately low entropy direct path + at least one support path, OR
            - Higher entropy direct path + multiple support paths (more paths needed as entropy increases), OR
            - No direct path but multiple low entropy 2-3 hop paths

            2. Score 4
            - Moderately low entropy direct path without support paths, OR
            - Relatively high entropy direct path + 1-2 support paths, OR
            - High entropy direct path + numerous support paths, OR
            - No direct path but moderately low entropy 2-3 hop paths

            3. Score 3
            - High entropy direct path, OR
            - 1-3 high entropy support paths

            4. Score 2
            - No direct path
            - Multiple high entropy support paths, none completely correct

            5. Score 1
            - No direct path
            - Very few support paths with errors, OR
            - Only 1-2 high entropy support paths with errors

            6. Score 0
            - No direct path
            - Irrelevant or unrelated paths
            - No logical connection to target triple

            For each evaluation, please provide:
            1. Analyze direct paths(It is very important to confirm whether a direct path exists, because an incorrect judgment will lead to a significant difference in the assigned score).
            2. Analyze support paths 
            3. Your reasoning for the assigned score
            4. The final score (0-5)

            {few_shot_example}


            ### Given Subgraph Triples:
            {facts_str}

            ### Target Triple:
            ({A}, {relation}, {B})

            ### Your Reasoning:
            Explain step-by-step how the given facts lead to (or fail to lead to) the target triple.

            Then output the score on a new line like:
            Final Confidence Score: <integer between 0 and 5>
            """

    return prompt

def evaluate_all_subgraphs(all_subgraphs, output_path, use_openai=True):
    """
    Evaluate all subgraphs using LLM and save results in one JSON file.
    If output_path already contains results, avoid re-evaluating existing triples.
    If a triple has an empty subgraph, automatically assign a score of 0.

    Args:
        all_subgraphs (dict): Dict with key=str(triple), value=list of subgraph triples
        output_path (str): File path to save combined evaluation result
        use_openai (bool): Whether to use OpenAI or DeepSeek

    Returns:
        None
    """
    # Load existing results if file already exists
    if os.path.exists(output_path):
        with open(output_path, "r", encoding="utf-8") as f:
            results = json.load(f)
    else:
        results = {}

    client = OpenAI(api_key=OPENAI_API_KEY)
    tokenizer = tiktoken.encoding_for_model("gpt-4")

    for i, (triple_key_str, subgraph_triples) in enumerate(all_subgraphs.items()):
        if triple_key_str in results:
            logging.info(f"[{i+1}/{len(all_subgraphs)}] Skipping already evaluated triple {triple_key_str}")
            continue
            
        # Check if subgraph is empty, assign score 0 directly if true
        if len(subgraph_triples) == 0:
            logging.info(f"[{i+1}/{len(all_subgraphs)}] Empty subgraph for {triple_key_str} → Auto-assigning Score: 0")
            results[triple_key_str] = {
                "score": 0,
                "reasoning": "Automatically assigned score 0 due to empty subgraph.",
                "num_edges": 0,
            }
            
            # Save updated result incrementally
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2)
            continue

        try:
            target_triple = eval(triple_key_str)

            # Prepare entropy-based subgraph format
            edges_with_entropy = []
            for triple in subgraph_triples:
                if len(triple) >= 4:
                    h, r, t = triple[0], triple[1], triple[2]
                    entropy = float(triple[3])
                    edges_with_entropy.append((h, r, t, entropy))
                else:
                    raise ValueError(f"Invalid triple format: {triple}")

            # Format prompt
            prompt = format_prompt_with_entropy(edges_with_entropy, target_triple)

            # Token length using GPT-4 tokenizer
            num_tokens = len(tokenizer.encode(prompt))
            print(f"[{i+1}/{len(all_subgraphs)}] Prompt token count: {num_tokens}")

            llm_response = ""
            while llm_response == "":
                # Get full LLM response
                response = client.chat.completions.create(
                    model="o4-mini",
                    messages=[
                        {"role": "system", "content": "You are an expert knowledge graph evaluator."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=1,
                    #max_completion_tokens=1024
                )

                llm_response = response.choices[0].message.content.strip()
                score = extract_confidence_score(llm_response)

                results[triple_key_str] = {
                    "score": score,
                    "reasoning": llm_response,
                    "num_edges": len(edges_with_entropy),
                }

            # Save updated result incrementally
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2)

            logging.info(f"[{i+1}/{len(all_subgraphs)}] Evaluated {triple_key_str} → Score: {score}")

        except Exception as e:
            logging.warning(f"Skipping triple {triple_key_str} due to error: {e}")

    logging.info(f"All judgments saved to {output_path}")


def process_unlearn_triples(json_path, entropy_threshold, output_path):
    """
    Process unlearn triples from a JSON file, create subgraphs, and evaluate them.
    
    Args:
        json_path (str): Path to the JSON file containing unlearn triples.
        entropy_threshold (float): Threshold for filtering triples based on entropy.
        output_path (str): Directory path to save the processed subgraphs.
        
    Returns:
        None
    """
    
    # Read the input JSON file
    with open(json_path, 'r', encoding='utf-8') as f:
        unlearn_triples = json.load(f)
    
    # Process triples and create subgraphs
    subgraphs = {}
    for triple_str, values in unlearn_triples.items():
        # Parse the triple string into head, relation, tail
        triple_tuple = eval(triple_str)
        head, relation, tail = triple_tuple
        
        # Extract values
        entropy, yes_prob, no_prob, unknown_prob = values
        
        # Create the triple with all information
        triple_info = [
            head,
            relation,
            tail,
            entropy,
            yes_prob,
            no_prob,
            unknown_prob
        ]
        
        # Create subgraph (either containing the triple or empty)
        if entropy <= entropy_threshold:
            subgraphs[triple_str] = [triple_info]
        else:
            subgraphs[triple_str] = []  # Empty subgraph if entropy exceeds threshold
    
    # Create the output filename
    base_name = os.path.basename(json_path)
    name_without_ext = os.path.splitext(base_name)[0]
    output_file = os.path.join(output_path, f"{name_without_ext}_subgraph.json")
    
    # Ensure the output directory exists
    os.makedirs(output_path, exist_ok=True)
    
    # Save the subgraphs to the output file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(subgraphs, f, indent=2)
    
    print(f"Subgraphs saved to {output_file}")
    
    # Load the saved subgraphs for evaluation
    with open(output_file, 'r', encoding='utf-8') as f:
        loaded_subgraphs = json.load(f)
    
    # Call evaluate_all_subgraphs with the loaded subgraphs
    evaluation_output = os.path.join(output_path, f"{name_without_ext}_evaluation.json")
    evaluate_all_subgraphs(loaded_subgraphs, evaluation_output, use_openai=True)
    
    print(f"Evaluation results saved to {evaluation_output}")