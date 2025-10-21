# Set up experiemnt

1. provide AWS ressources with provided access
```bash
export AWS_DEFAULT_REGION="..."
export AWS_ACCESS_KEY_ID="..."
export AWS_SECRET_ACCESS_KEY="..."
export AWS_SESSION_TOKEN="..."
```
or with the console : Select Region Orgeon -> Amazon Sagemaker AI -> Notebook -> Create -> Choose g6.2xlarge -> Specify Amazon Linux2023 -> Add GitRef

2. Fork and clone git repo, setup env and boltz install

```bash
git clone YOUR_FORKED_REPO_URL
cd <name_of_your_fork>
conda env create -f environment.yml --name boltz
conda activate boltz
pip install -e ".[cuda]"
```

3. prepare the data to use for test

```bash
wget https://d2v9mdonbgo0hk.cloudfront.net/hackathon_data.tar.gz
mkdir hackathon_data
tar -xvf hackathon_data.tar.gz -C hackathon_data
```


3. run the evaluatuion for a baseline test and inspect [using abag dataset c.a. protein protein complex]

```bash
 python hackathon/predict_hackathon.py \
     --input-jsonl hackathon_data/datasets/abag_public/abag_public.jsonl \
     --msa-dir hackathon_data/datasets/abag_public/msa/ \
     --submission-dir ./my_predictions \
     --intermediate-dir ./tmp/ \
     --result-folder ./my_results
```

4. analyse first results


# Modify the scripts

1. the ranking script to promote prediction without overfitting on 1 metric. Instead of using only iptm, we combine multiple confidence metrics:

```python
for prediction_dir in prediction_dirs:
    config_pdbs = sorted(prediction_dir.glob(f"{datapoint.datapoint_id}_config_*_model_*.pdb"))
    for pdb_path in config_pdbs:
        confidence_path = pdb_path.parent / f"confidence_{pdb_path.stem}.json"
        score = 0.0
        
        if confidence_path.exists():
            with open(confidence_path) as f:
                conf_data = json.load(f)
                iptm = conf_data.get("root", {}).get("iptm", 0)
                ptm = conf_data.get("root", {}).get("ptm", 0)
                plddt_mean = conf_data.get("plddt", {}).get("mean", 0)

                # Weighted ensemble score
                score = (
                    0.5 * iptm +        # interface quality
                    0.3 * (ptm / 100) + # overall fold
                    0.2 * (plddt_mean / 100)  # local confidence
                )
        else:
            score = 0.0
        
        pdb_scores.append((score, pdb_path))

# Sort by composite score
pdb_scores.sort(key=lambda x: x[0], reverse=True)
```


2. the preparation of yaml for structure prediction.
* We use Higher temperature for diversity combined to Recycling for better structure estimation and also Potentials because we don't know yet if its rlevant

```python
configs.append((
    input_dict,
    ["--diffusion_samples", "5","--step_scale", "0.8", "--recycling_steps", "3", "--use_potentials"]
))
```
* we try setting biologically meaningful CDR constraints by identifying exposed (hydrophilic) residue on the anigen and Real CDR Regions (Loops)
```python
        modified_dict = input_dict.copy()
        modified_dict["constraints"] = [
            {
                "contact": {
                    "token1": ["H", h3_res],
                    "token2": ["A", pos],
                    "distance": 4.0,
                    "std": 1.5
                }
            },
            {
                "contact": {
                    "token1": ["L", l3_res],
                    "token2": ["A", pos],
                    "distance": 4.0,
                    "std": 1.5
                }
            }
        ]
```

3. evaluate
```bash
python hackathon/evaluate_abag.py \
    --dataset-file hackathon_data/datasets/abag_public/abag_public.jsonl \
    --submission-folder my_predictions \
    --result-folder ./abag_public_evaluation/
```

4. add and commit changes

5. submit report and link to forked updated repo

   
                     