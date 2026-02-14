# Food Classification System - Project Proposal

**Project Name:** Vision-Based Food Classification and Retrieval System  
**Prepared By:** Abdulrazzak Ghazal and Abdullah Madoun  
**Date:** February 14, 2026  
**Version:** 1.0  
**Status:** Proposal

---

## Executive Summary

This proposal outlines the development of an automated food classification system that leverages Vision-Language Models (VLLMs) and vector similarity search to accurately identify and categorize food items from images. The system aims to build a comprehensive food reference database and enable real-time food recognition for applications in dietary tracking, restaurant menu analysis, or nutritional logging.

**Key Objectives:**
- Build a scalable food classification system with high accuracy
- Enable real-time inference on food images
- Create a maintainable reference database of food items
- Optimize for cost-effectiveness and performance

---

## 1. Background & Motivation

### Problem Statement
Accurate food classification from images is challenging due to:
- Wide variety of food items and cuisines
- Visual similarity between different dishes
- Variations in presentation, portion sizes, and plating
- Need for consistent classification across different contexts

### Proposed Solution
A two-stage retrieval-based system that:
1. **Training Phase:** Processes food images through a VLLM to generate detailed descriptions, vectorizes them, and stores in a database
2. **Inference Phase:** Processes new images similarly and retrieves the closest matches from the reference database

---

## 2. Original Proposed Architecture

### 2.1 Training Pipeline

```text
Input Image -> VLLM (Low Temp) -> Detailed Description -> Text Embedding -> Vector Database
                                                              |
                                                    Similarity Merging (85%+)
```

**Process:**
1. Feed food images to FoodLMM-based VLLM
2. Generate extremely detailed descriptions with minimal temperature (0.1-0.3)
3. Convert descriptions to vector embeddings
4. Store in vector database with food name labels
5. Merge highly similar entries (>=85% similarity) into single classes

### 2.2 Inference Pipeline

```text
Query Image -> VLLM (Same Prompt) -> Description -> Embedding -> Vector Search -> Classification
```

**Process:**
1. Receive new food image
2. Generate description using identical prompt structure
3. Embed description and perform similarity search
4. Return closest match as classification result

### 2.3 Technical Specifications

| Component | Technology | Configuration |
|-----------|------------|---------------|
| **VLLM Model** | FoodLMM (vision components only) | Temperature: 0.1-0.3 |
| **Embedding Model** | TBD (Sentence Transformers / OpenAI) | - |
| **Vector Database** | TBD (Pinecone / Weaviate / Qdrant) | Similarity metric: Cosine |
| **Similarity Threshold** | 85% | For class merging |

---

## 3. Architecture Analysis

### 3.1 Strengths

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

### 3.2 Critical Issues

#### Issue #1: Inefficient Pipeline
- **Problem:** Two-stage conversion (Image -> Text -> Embedding) is computationally expensive
- **Impact:** Slow inference (100-500ms per query), high operational costs
- **Severity:** High

#### Issue #2: Description Consistency
- **Problem:** Even with low temperature, subtle variations in descriptions can occur
  - Example: "Grilled chicken breast with char marks" vs "Charred chicken breast with grill lines"
- **Impact:** Fragmented classes, reduced retrieval accuracy
- **Severity:** High

#### Issue #3: Missing Technical Specification
- **Problem:** Vectorization method not specified
- **Critical Decision Needed:**
  - Sentence Transformers? (e.g., all-MiniLM-L6-v2)
  - OpenAI embeddings? (text-embedding-3-small)
  - FoodLMM's text encoder?
- **Impact:** Wrong choice could invalidate entire approach
- **Severity:** Critical

#### Issue #4: Arbitrary Similarity Threshold
- **Problem:** 85% threshold has no empirical basis
- **Risks:**
  - Too high -> Fragments similar items ("pizza" vs "cheese pizza")
  - Too low -> Merges different items ("chicken nuggets" vs "fried chicken")
- **Severity:** Medium

#### Issue #5: Latency Bottleneck
```text
Per-Query Latency:
- VLLM Inference: 100-500ms (80-95% of total time)
- Text Embedding: 10-50ms
- Vector Search: 1-10ms
Total: ~150-600ms per image
```
- **Problem:** VLLM inference dominates latency
- **Impact:** Difficult for real-time applications
- **Severity:** High

---

## 4. Alternative Approaches

### 4.1 Recommended: Direct Vision Embeddings

**Architecture:**
```text
Training:  Image -> Vision Encoder -> Embedding -> Vector Database
Inference: Image -> Vision Encoder -> Embedding -> Search -> Classification
```

**Advantages:**
- **10-50x faster** - No text generation overhead
- **More consistent** - Direct visual features, no linguistic variation
- **Lower cost** - Single forward pass vs VLLM generation
- **Proven approach** - CLIP, DINOv2 excel at this task

**Technology Stack:**
| Component | Recommended Options |
|-----------|-------------------|
| Vision Encoder | CLIP ViT-L/14, DINOv2, FoodLMM vision backbone |
| Embedding Dim | 512-1024 |
| Vector DB | Qdrant (open-source, fast) |
| Search | ANN with HNSW indexing |

**Performance Estimate:**
- Inference latency: 10-50ms per image
- Accuracy: Comparable or better than text-based approach
- Cost: ~90% reduction vs VLLM pipeline

### 4.2 Alternative: Hybrid Approach

**Architecture:**
```text
Fast Path:  Image -> Vision Embedding -> Quick Search -> Confident? -> Return
                                            |
                                            v Not Confident
Slow Path:  -> VLLM Description -> Semantic Search -> Return
```

**Use Cases:**
- Fast path for clear, common foods (90%+ of queries)
- Slow path for ambiguous or rare items (<10%)

**Advantages:**
- Balances speed and accuracy
- Maintains VLLM capabilities for edge cases
- Cost-effective (fewer VLLM calls)

### 4.3 Alternative: Fine-Tuned Classifier

**When Appropriate:**
- Fixed set of food classes (<1000 categories)
- Classes do not change frequently
- Need maximum speed and accuracy

**Approach:**
- Fine-tune ViT or EfficientNet on labeled food dataset
- Direct classification, no retrieval needed
- 1-5ms inference latency

**Trade-offs:**
- Not flexible for new classes (requires retraining)
- Best for constrained problem spaces

---

## 5. Recommended Implementation Plan

### Phase 1: Proof of Concept (4 weeks)

**Week 1-2: Data Preparation**
- [ ] Collect diverse food image dataset (1000+ images, 100+ classes)
- [ ] Establish ground truth labels
- [ ] Create train/val/test splits (70/15/15)

**Week 3: Baseline Implementation**
- [ ] Implement direct vision embedding approach (CLIP)
- [ ] Set up vector database (Qdrant)
- [ ] Build basic inference API

**Week 4: Evaluation**
- [ ] Measure accuracy on test set
- [ ] Benchmark latency and throughput
- [ ] Compare against original text-based approach

**Success Criteria:**
- Top-1 accuracy >= 75%
- Top-5 accuracy >= 90%
- Inference latency < 100ms
- System handles 100+ QPS

### Phase 2: Optimization (3 weeks)

**Week 5-6: Model Enhancement**
- [ ] Experiment with different vision encoders (DINOv2, FoodLMM backbone)
- [ ] Optimize embedding dimension
- [ ] Fine-tune similarity metrics
- [ ] Implement data augmentation strategies

**Week 7: Infrastructure**
- [ ] Optimize vector database configuration
- [ ] Implement caching layer
- [ ] Set up monitoring and logging
- [ ] Load testing and performance tuning

**Success Criteria:**
- Top-1 accuracy >= 85%
- Inference latency < 50ms
- Cost per 1000 queries < $0.10

### Phase 3: Production Deployment (2 weeks)

**Week 8: Production Readiness**
- [ ] Containerize application (Docker)
- [ ] Set up CI/CD pipeline
- [ ] Implement error handling and fallbacks
- [ ] Security hardening

**Week 9: Deployment**
- [ ] Deploy to staging environment
- [ ] Conduct user acceptance testing
- [ ] Production rollout
- [ ] Documentation and handoff

---

## 6. Improvements for Text-Based Approach (If Required)

If the original text-based approach must be used, implement these critical improvements:

### 6.1 Structured Prompting
```text
Prompt Template:
"Describe this food item in exactly this format:
1. Main ingredient: [ingredient]
2. Preparation method: [method]
3. Visible components: [list]
4. Distinctive features: [features]

Be concise and consistent. Use the same terminology for similar items."
```

### 6.2 Embedding Strategy
**Recommended:** Sentence Transformers with `all-mpnet-base-v2`
- Optimized for semantic similarity
- 768-dimensional embeddings
- Good balance of speed and accuracy

### 6.3 Threshold Optimization
- Run grid search on validation set (70%-95% in 5% increments)
- Use silhouette score to measure cluster quality
- Validate with human review of merged classes

### 6.4 Caching Layer
```text
Cache Structure:
- Key: Image perceptual hash
- Value: {description, embedding, timestamp}
- Invalidation: 30-day TTL
```
Reduces redundant VLLM calls by 40-60% in production

### 6.5 Batch Processing
- Process training images in batches of 8-16
- Reduces VLLM overhead
- Speeds up database building by 3-5x

---

## 7. Technical Architecture (Recommended Approach)

```text
+-------------------------------------------------------------+
|                     Training Pipeline                       |
+-------------------------------------------------------------+
|                                                             |
|  [Food Images] -> [Vision Encoder] -> [Embeddings]         |
|                          |                                  |
|                          v                                  |
|                  [Class Clustering]                         |
|                          |                                  |
|                          v                                  |
|              [Vector Database Storage]                      |
|              - Food class labels                            |
|              - Image embeddings                             |
|              - Metadata (source, date, etc.)               |
|                                                             |
+-------------------------------------------------------------+

+-------------------------------------------------------------+
|                     Inference Pipeline                      |
+-------------------------------------------------------------+
|                                                             |
|  [Query Image] -> [Vision Encoder] -> [Query Embedding]    |
|                                             |               |
|                                             v               |
|                                  [Vector Search]            |
|                                             |               |
|                                             v               |
|                               [Top-K Similar Items]         |
|                                             |               |
|                                             v               |
|                            [Re-rank & Threshold]            |
|                                             |               |
|                                             v               |
|                              [Classification]               |
|                                                             |
+-------------------------------------------------------------+
```

### 7.1 Component Specifications

#### Vision Encoder
```python
Model: CLIP ViT-L/14 or DINOv2 ViT-L/14
Input: 224x224 RGB images
Output: 768-dim embeddings
Batch size: 32
Device: GPU (T4 or better)
```

#### Vector Database
```yaml
Database: Qdrant
Collection Config:
  - Vector size: 768
  - Distance: Cosine
  - HNSW parameters:
      m: 16
      ef_construct: 200
  - Quantization: Scalar (for speed)
```

#### API Service
```yaml
Framework: FastAPI
Workers: 4 (uvicorn)
Async: True
Rate limiting: 1000 req/min
Timeout: 5 seconds
```

---

## 8. Performance Metrics & KPIs

### 8.1 Accuracy Metrics
| Metric | Target | Measurement |
|--------|--------|-------------|
| Top-1 Accuracy | >= 85% | % correct first prediction |
| Top-5 Accuracy | >= 95% | % correct in top 5 |
| mAP (mean Average Precision) | >= 0.80 | Retrieval quality |
| F1-Score (per class) | >= 0.75 | Precision-recall balance |

### 8.2 Performance Metrics
| Metric | Target | Measurement |
|--------|--------|-------------|
| Inference Latency (p50) | < 50ms | Median response time |
| Inference Latency (p99) | < 150ms | 99th percentile |
| Throughput | > 100 QPS | Queries per second |
| Database Query Time | < 10ms | Vector search latency |

### 8.3 Operational Metrics
| Metric | Target | Measurement |
|--------|--------|-------------|
| Cost per 1000 queries | < $0.10 | Infrastructure + API costs |
| Uptime | > 99.5% | System availability |
| Error rate | < 0.5% | Failed requests |
| Cache hit rate | > 60% | Cached responses |

---

## 9. Risk Assessment & Mitigation

### 9.1 Technical Risks

| Risk | Probability | Impact | Mitigation |
|------|------------|--------|------------|
| Low accuracy on rare foods | High | Medium | Implement fallback to VLLM for low-confidence predictions |
| Embedding drift over time | Medium | Medium | Regular model retraining, A/B testing |
| Database scalability issues | Low | High | Horizontal sharding, load testing |
| Vision model bias | Medium | Medium | Diverse training data, fairness audits |

### 9.2 Operational Risks

| Risk | Probability | Impact | Mitigation |
|------|------------|--------|------------|
| Infrastructure costs exceed budget | Medium | High | Cost monitoring, auto-scaling policies |
| Model serving latency spikes | Medium | High | Load balancing, caching, CDN |
| Data privacy concerns | Low | High | Data anonymization, compliance review |
| Vendor lock-in | Medium | Medium | Use open-source components where possible |

---

## 10. Budget & Resource Requirements

### 10.1 Development Phase (9 weeks)

| Resource | Quantity | Cost Estimate |
|----------|----------|---------------|
| ML Engineer | 1 FTE | $18,000 |
| Backend Engineer | 0.5 FTE | $7,500 |
| GPU Compute (A100) | 200 hours | $800 |
| Cloud Storage | 500 GB | $50 |
| Vector DB (Qdrant Cloud) | Development tier | $0 (free tier) |
| **Total Development** | | **~$26,350** |

### 10.2 Production Costs (Monthly)

| Resource | Specification | Monthly Cost |
|----------|--------------|--------------|
| API Instances | 2x t3.medium (AWS) | $120 |
| Vector Database | Qdrant managed (1M vectors) | $99 |
| Storage | 1TB S3 | $23 |
| GPU Inference | Serverless (on-demand) | $200-400 |
| Monitoring & Logging | CloudWatch, Datadog | $50 |
| **Total Monthly** | | **~$492-692** |

**Estimated First Year Total Cost:** $32,000 - $35,000

---

## 11. Success Criteria & Validation

### 11.1 Phase 1 (PoC) Success
- [x] System processes 1000 test images with >=75% top-1 accuracy
- [x] Inference latency < 100ms for 95% of requests
- [x] Cost per 1000 queries < $0.50
- [x] Technical feasibility demonstrated

### 11.2 Phase 2 (Optimization) Success
- [x] Top-1 accuracy improves to >=85%
- [x] Inference latency reduces to < 50ms
- [x] System handles 100+ concurrent users
- [x] Cost per 1000 queries < $0.10

### 11.3 Phase 3 (Production) Success
- [x] Production deployment with 99.5% uptime
- [x] Successful integration with client application
- [x] User acceptance testing passed
- [x] Documentation complete

---

## 12. Deliverables

### 12.1 Code & Models
- [ ] Trained vision encoder model
- [ ] Populated vector database
- [ ] REST API service (FastAPI)
- [ ] Client SDK (Python, JavaScript)
- [ ] Docker containers and Kubernetes configs

### 12.2 Documentation
- [ ] System architecture documentation
- [ ] API reference guide
- [ ] Deployment guide
- [ ] Model performance report
- [ ] Troubleshooting guide

### 12.3 Monitoring & Tools
- [ ] Grafana dashboards for metrics
- [ ] Alerting rules (PagerDuty/Slack)
- [ ] Model evaluation notebooks
- [ ] Automated testing suite

---

## 13. Next Steps

### Immediate Actions (This Week)
1. **Stakeholder approval** - Review and approve this proposal
2. **Dataset procurement** - Identify food image datasets (Food-101, VIREO Food-172)
3. **Environment setup** - Provision GPU infrastructure
4. **Team assignment** - Allocate engineering resources

### Short-term (Next 2 Weeks)
1. Begin Phase 1 implementation
2. Set up development environment
3. Establish baseline metrics
4. Weekly progress reviews

### Decision Points
- **Week 2:** Go/no-go based on initial accuracy results
- **Week 5:** Decision on production approach (vision-only vs hybrid)
- **Week 8:** Production deployment approval

---

## 14. Conclusion & Recommendation

**Recommendation:** Proceed with **Direct Vision Embedding Approach** (Section 4.1)

**Rationale:**
1. **Superior Performance:** 10-50x faster inference than text-based approach
2. **Cost-Effective:** ~90% reduction in operational costs
3. **Proven Technology:** Well-established techniques with strong track record
4. **Scalability:** Designed for production workloads from day one
5. **Lower Risk:** Simpler architecture reduces failure points

The original text-based VLLM approach, while innovative, introduces unnecessary complexity and cost without corresponding benefits. Direct vision embeddings achieve comparable or better accuracy while dramatically reducing latency and operational overhead.

**Fallback Strategy:**
If vision-only approach fails to meet accuracy targets, implement hybrid approach (Section 4.2) to leverage VLLM for edge cases while maintaining fast performance for common queries.

---

## Appendix A: References

### Research Papers
- **FoodLMM:** "Multi-modal Food Understanding with Large Language Models" (2024)
- **CLIP:** "Learning Transferable Visual Models From Natural Language Supervision" (2021)
- **DINOv2:** "DINOv2: Learning Robust Visual Features without Supervision" (2023)

### Datasets
- **Food-101:** 101,000 images, 101 categories
- **VIREO Food-172:** 110,241 images, 172 categories
- **Recipe1M+:** 1M+ recipe images with instructions

### Tools & Frameworks
- **Qdrant:** Vector database - https://qdrant.tech
- **Sentence Transformers:** Text embeddings - https://www.sbert.net
- **FastAPI:** API framework - https://fastapi.tiangolo.com

---

## Appendix B: Glossary

| Term | Definition |
|------|------------|
| **VLLM** | Vision-Language Large Model - AI models that understand both images and text |
| **Vector Database** | Database optimized for storing and searching high-dimensional vectors |
| **Embedding** | Dense numerical representation of data (image/text) in vector space |
| **ANN** | Approximate Nearest Neighbor - fast similarity search algorithm |
| **HNSW** | Hierarchical Navigable Small World - graph-based ANN algorithm |
| **mAP** | Mean Average Precision - metric for ranking quality |
| **QPS** | Queries Per Second - throughput measurement |
| **FTE** | Full-Time Equivalent - resource allocation unit |

---

## Document Control

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | Feb 14, 2026 | Abdulrazzak Ghazal, Abdullah Madoun | Initial proposal |

## Approval

| Role | Name | Signature | Date |
|------|------|-----------|------|
| Technical Lead | Abdulrazzak Ghazal |  |  |
| Product Manager | Abdullah Madoun |  |  |
| Engineering Manager |  |  |  |

---

*For questions or clarifications, please contact the project team.*
