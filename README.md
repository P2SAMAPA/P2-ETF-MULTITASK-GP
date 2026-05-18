# Multi‑task Gaussian Process Engine

Implements a multi‑task Gaussian Process (MTGP) with a macro‑conditioned kernel for joint prediction of ETF returns. Uses inducing points for scalability. Outputs the predictive mean as the score (expected return). Multi‑window evaluation selects the best window per ETF.

- **Multi‑task kernel:** RBF (time) + coregionalization (tasks)
- **Inducing points:** 100 (configurable)
- **Training:** Exact GP with 500 iterations
- **Windows:** 63, 252, 504, 1008, 2016 days (best per ETF)
- **Output:** top 3 ETFs per universe by predictive mean

Runs daily on GitHub Actions.

## Local execution

```bash
pip install -r requirements.txt
export HF_TOKEN=<your_token>
python trainer.py
streamlit run streamlit_app.py
