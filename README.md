# Food Classification System - Project Proposal

**Course Project Proposal**  
**Date:** February 14, 2026  

---

## Executive Summary

This project explores the development of an automated food classification system using Vision-Language Models (VLLMs) and vector similarity search. The goal is to investigate the feasibility and effectiveness of using text descriptions as an intermediate representation for food image classification.

**Project Objectives:**
- Implement and evaluate a VLLM-based food classification approach
- Compare different architectural strategies (text-based vs. direct vision embeddings)
- Analyze trade-offs in accuracy, speed, and implementation complexity
- Provide recommendations based on experimental results

---

## 1. Background & Motivation

Accurate food classification from images is challenging due to visual similarity between dishes, variations in presentation and lighting, and the need for consistent classification. This project investigates whether using Vision-Language Models to generate text descriptions as an intermediate representation can improve classification performance compared to direct vision-based approaches.

---

## 2. Original Proposed Architecture

### 2.1 Training Pipeline

```
Input Image → VLLM (Low Temp) → Detailed Description → Text Embedding → Vector Database
                                                              ↓
                                                    Similarity Merging (85%+)
```

**Process:**
1. Feed food images to FoodLMM-based VLLM
2. Generate extremely detailed descriptions with minimal temperature (0.1-0.3)
3. Convert descriptions to vector embeddings
4. Store in vector database with food name labels
5. Merge highly similar entries (≥85% similarity) into single classes

### 2.2 Inference Pipeline

```
Query Image → VLLM (Same Prompt) → Description → Embedding → Vector Search → Classification
```

**Process:**
1. Receive new food image
2. Generate description using identical prompt structure
3. Embed description and perform similarity search
4. Return closest match as classification result

## 2. Proposed Architecture (Original Approach)

### 2.1 Training Pipeline

```
Input Image → VLLM (Low Temp) → Detailed Description → Text Embedding → Vector Database
                                                              ↓
                                                    Similarity Merging (85%+)
```

**Process:**
1. Feed food images to a Vision-Language Model (considering FoodLMM)
2. Generate detailed descriptions with low temperature (0.1-0.3) for consistency
3. Convert descriptions to vector embeddings
4. Store in vector database
5. Merge highly similar entries (≥85% similarity) into single classes

### 2.2 Inference Pipeline

```
Query Image → VLLM (Same Prompt) → Description → Embedding → Vector Search → Classification
```

### 2.3 Key Questions to Investigate

- Which embedding model to use? (Sentence Transformers vs OpenAI vs model's text encoder)
- What is the optimal similarity threshold?
- How much does description variance affect accuracy?
- Is the two-stage pipeline worth the computational cost?

---

## 3. Architecture Analysis

### 3.1 Strengths ✅

1. **Leverages Powerful Models**
   - FoodLMM has strong food-specific understanding
   - Can capture nuanced visual differences

2. **Deterministic Descriptions**
   - Low temperature ensures consistent outputs
   - Reduces randomness in classification

3. **Handles Variations**
   - Similarity merging consolidates near-duplicates
   - Flexible to different presentations of same food

4. **Scalable Retrieval**
   - Vector search is fast once database is built
   - Can handle large food catalogs

### 3.2 Potential Issues

#### Issue #1: Computational Inefficiency
- Two-stage conversion (Image → Text → Embedding) is slower than direct vision embeddings
- Each inference requires a full VLLM forward pass
- Estimated latency: 100-500ms per image vs 10-50ms for direct vision approaches

#### Issue #2: Description Consistency
- Even with low temperature, descriptions may vary slightly
- Example: "Grilled chicken with char marks" vs "Charred chicken with grill lines"
- Could fragment similar items into different classes

#### Issue #3: Embedding Choice
**Critical decision needed:** Which text embedding model?
- Sentence Transformers (e.g., all-MiniLM-L6-v2)
- OpenAI embeddings (text-embedding-3-small)
- VLLM's own text encoder

#### Issue #4: Arbitrary Threshold
- 85% similarity threshold has no empirical basis
- Risk of fragmenting similar items (too high) or merging different items (too low)
- Needs experimental validation

---

## 4. Alternative Approaches to Evaluate

### 4.1 Baseline: Direct Vision Embeddings

**Architecture:**
```
Training:  Image → Vision Encoder (CLIP/DINOv2) → Embedding → Vector Database
Inference: Image → Vision Encoder → Embedding → Search → Classification
```

**Why evaluate this:**
- Industry-standard approach
- 10-50x faster than text-based pipeline
- Provides baseline to compare against
- More consistent representations (no linguistic variation)

**Possible models:**
- CLIP (ViT-B/32 or ViT-L/14)
- DINOv2
- FoodLMM's vision backbone only

### 4.2 Hybrid Approach (Optional)

If time permits, could explore:
```
Quick Path:  Image → Vision Embedding → Search (if confident) → Return
Slow Path:   → VLLM Description → Search (if uncertain) → Return
```

This balances speed (vision) with semantic understanding (VLLM) for edge cases.

---

## 5. Project Plan

### Timeline: 8-10 weeks

**Weeks 1-2: Setup & Data Preparation**
- Set up development environment (Google Colab or local GPU)
- Download public food dataset (Food-101 or subset)
- Split data: 70% train, 15% validation, 15% test
- Explore 50-100 food categories to keep scope manageable

**Weeks 3-4: Implement Baseline**
- Implement direct vision embedding approach (CLIP)
- Set up simple vector database (FAISS or Qdrant)
- Build basic inference pipeline
- Measure baseline accuracy and speed

**Weeks 5-6: Implement Text-Based Approach**
- Choose VLLM (could use open-source alternatives if FoodLMM unavailable)
- Generate descriptions with different temperature settings
- Test different text embedding models
- Experiment with similarity thresholds (70%, 80%, 85%, 90%)

**Weeks 7-8: Evaluation & Analysis**
- Compare approaches on same test set
- Measure: accuracy, inference time, memory usage
- Analyze failure cases
- Document findings

**Weeks 9-10: Report & Presentation**
- Write final report
- Create presentation slides
- Prepare demo (if possible)

---

## 6. Implementation Considerations

### 6.1 For Text-Based Approach

**Structured Prompting:**
```
Template: "Describe this food item using:
- Main ingredient: [ingredient]
- Preparation: [method]
- Visible components: [list]
Keep descriptions consistent and concise."
```

**Embedding Model Options:**
- Sentence Transformers: `all-MiniLM-L6-v2` or `all-mpnet-base-v2`
- Can experiment with different models and compare

**Threshold Selection:**
- Test range: 70%, 75%, 80%, 85%, 90%
- Use validation set to determine optimal value
- Manual inspection of merged classes

### 6.2 For Vision-Based Approach

**Model Selection:**
- CLIP: Pre-trained, good zero-shot performance
- Could try different architectures (ViT-B/32 vs ViT-L/14)

**Vector Database:**
- FAISS (simple, local, good for projects)
- Qdrant (if want to learn more complex system)

### 6.3 Evaluation Metrics

Focus on these key metrics:
- **Top-1 Accuracy:** Percentage of correct first predictions
- **Top-5 Accuracy:** Percentage correct in top 5 results
- **Inference Time:** Average time per image
- **Confusion Analysis:** Which foods get confused most?

---

## 7. System Architecture Overview

### Approach A: Text-Based (Original)
```
┌─────────────────────────────────────────┐
│         Training Phase                   │
├─────────────────────────────────────────┤
│                                          │
│  Images → VLLM → Descriptions           │
│             ↓                            │
│        Text Embeddings                   │
│             ↓                            │
│      Vector Database                     │
│                                          │
└─────────────────────────────────────────┘

┌─────────────────────────────────────────┐
│         Inference Phase                  │
├─────────────────────────────────────────┤
│                                          │
│  Query Image → VLLM → Description       │
│                  ↓                       │
│            Text Embedding                │
│                  ↓                       │
│           Vector Search                  │
│                  ↓                       │
│            Classification                │
│                                          │
└─────────────────────────────────────────┘
```

### Approach B: Vision-Based (Baseline)
```
┌─────────────────────────────────────────┐
│         Training Phase                   │
├─────────────────────────────────────────┤
│                                          │
│  Images → Vision Encoder (CLIP)         │
│                  ↓                       │
│          Image Embeddings                │
│                  ↓                       │
│         Vector Database                  │
│                                          │
└─────────────────────────────────────────┘

┌─────────────────────────────────────────┐
│         Inference Phase                  │
├─────────────────────────────────────────┤
│                                          │
│  Query Image → Vision Encoder           │
│                  ↓                       │
│           Image Embedding                │
│                  ↓                       │
│            Vector Search                 │
│                  ↓                       │
│            Classification                │
│                                          │
└─────────────────────────────────────────┘
```

---

## 8. Expected Outcomes & Evaluation

### 8.1 Comparison Metrics

Will compare both approaches on:

| Metric | Why It Matters |
|--------|----------------|
| **Top-1 Accuracy** | How often the top prediction is correct |
| **Top-5 Accuracy** | How often correct answer is in top 5 |
| **Average Inference Time** | Speed comparison |
| **Confusion Patterns** | Which foods get mixed up |

### 8.2 Research Questions to Answer

1. Does using text descriptions improve accuracy over direct vision embeddings?
2. How much does description consistency affect results?
3. What is the optimal similarity threshold for merging classes?
4. Is the computational overhead of VLLM justified by accuracy gains?
5. Which embedding model works best for food descriptions?

### 8.3 Expected Findings

**Hypothesis:** Direct vision embeddings will likely outperform text-based approach because:
- No information loss in image→text conversion
- More consistent representations
- Faster inference

**However,** text-based approach might excel at:
- Distinguishing visually similar foods with different preparation methods
- Providing interpretable reasons for classifications
- Zero-shot recognition of new food types

---

## 9. Potential Challenges

### 9.1 Technical Challenges

**Data Quality**
- Public food datasets may have inconsistent labels
- Images vary in quality and angle
- Some categories may overlap (e.g., "sandwich" vs "burger")

**Computational Resources**
- VLLM inference requires GPU access (Google Colab Pro or university cluster)
- May need to use smaller models if resources limited
- Vector database storage for large datasets

**Model Access**
- FoodLMM may not be publicly available or easy to set up
- May need to use alternative VLLMs (LLaVA, BLIP-2, etc.)
- API costs if using commercial models

### 9.2 Mitigation Strategies

- Start with smaller dataset (Food-101 subset, ~50 classes)
- Use free tier of vector databases (Qdrant Cloud, Pinecone)
- Leverage Google Colab for GPU access
- Use open-source models from Hugging Face
- Batch processing to reduce API calls

---

## 10. Resources Needed

### 10.1 Computational Resources
- GPU access (Google Colab Pro or university compute cluster)
- Storage: ~10-50GB for dataset and models
- RAM: 16GB+ recommended

### 10.2 Software/Tools
- Python 3.8+
- PyTorch or TensorFlow
- CLIP (from OpenAI)
- Sentence Transformers
- Vector database: FAISS (local) or Qdrant (free tier)
- Jupyter notebooks for experiments

### 10.3 Datasets
- Food-101 (public, free)
- Or VIREO Food-172
- Start with subset to stay within computational limits

---

## 11. Success Criteria

### Project will be considered successful if:

✅ **Both approaches implemented and tested**
- Text-based VLLM approach working
- Vision-based baseline working
- Fair comparison on same dataset

✅ **Comprehensive evaluation completed**
- Accuracy metrics calculated
- Speed comparison documented
- Failure analysis conducted

✅ **Clear findings documented**
- Advantages/disadvantages of each approach identified
- Recommendations supported by data
- Lessons learned documented

✅ **Deliverables completed**
- Code repository with documentation
- Final report with results
- Presentation ready

---

## 12. Project Deliverables

### 12.1 Code
- [ ] Python implementation of both approaches
- [ ] Jupyter notebooks for experiments
- [ ] Scripts for data preparation
- [ ] README with setup instructions
- [ ] GitHub repository

### 12.2 Documentation
- [ ] Final project report (10-15 pages)
  - Introduction & motivation
  - Related work / literature review
  - Methodology
  - Experimental results
  - Analysis & discussion
  - Conclusions & future work
  
- [ ] Code documentation (docstrings, comments)

### 12.3 Presentation
- [ ] Slide deck (15-20 slides)
- [ ] Demo (if time permits)
- [ ] Results visualization (confusion matrices, charts)

---

## 13. Conclusion

This project will investigate whether using Vision-Language Models to generate text descriptions as an intermediate representation can improve food classification accuracy compared to direct vision embeddings. 

**Main Research Question:** Is the computational overhead of generating text descriptions justified by accuracy improvements?

**Expected Contribution:**
- Empirical comparison of text-based vs vision-based food classification
- Analysis of trade-offs between interpretability and efficiency
- Documentation of lessons learned for future work

**Personal Learning Goals:**
- Gain hands-on experience with VLLMs and vision models
- Learn vector similarity search and embedding techniques
- Understand practical considerations in ML system design
- Develop skills in experimental design and evaluation

---

## Appendix A: Useful Resources

### Research Papers
- **CLIP:** "Learning Transferable Visual Models From Natural Language Supervision" (Radford et al., 2021)
- **DINOv2:** "DINOv2: Learning Robust Visual Features without Supervision" (Oquab et al., 2023)
- **FoodLMM:** Search for recent food-specific vision-language models

### Datasets
- **Food-101:** 101,000 images, 101 categories - https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/
- **VIREO Food-172:** 110,241 images, 172 categories

### Tools & Libraries
- **CLIP:** https://github.com/openai/CLIP
- **Sentence Transformers:** https://www.sbert.net
- **FAISS:** https://github.com/facebookresearch/faiss (vector search)
- **Qdrant:** https://qdrant.tech (vector database)
- **Hugging Face:** https://huggingface.co (pre-trained models)

### Tutorials
- Vector similarity search basics
- Fine-tuning vision models
- Working with embeddings

---

## Appendix B: Technical Terms

| Term | Definition |
|------|------------|
| **VLLM** | Vision-Language Model - AI that understands images and text |
| **Embedding** | Dense vector representation of data |
| **Vector Database** | Database for storing and searching vectors |
| **Similarity Search** | Finding closest vectors to a query |
| **Top-K Accuracy** | Percentage where correct answer is in top K predictions |
| **Zero-shot** | Model performs task without specific training |

---

**Note:** This is a living document and may be updated as the project progresses.
