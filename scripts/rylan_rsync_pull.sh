#rsync -avh rylansch@openmind-dtn.mit.edu:/om2/user/rylansch/FieteLab-Recursive-Nonstationary-CRP/data/arxiv_2022/arxiv-metadata-trimmed.csv data/arxiv_2022/arxiv-metadata-trimmed.csv
rsync -avh --exclude='*.joblib' rylansch@openmind-dtn.mit.edu:/om2/user/rylansch/FieteLab-Recursive-Nonstationary-CRP/00_prior/results/ 00_prior/results/
rsync -avh --exclude='*.joblib' rylansch@openmind-dtn.mit.edu:/om2/user/rylansch/FieteLab-Recursive-Nonstationary-CRP/01_mixture_of_gaussians/results/ 01_mixture_of_gaussians/results/
#rsync -avh --exclude='*.joblib' rylansch@openmind-dtn.mit.edu:/om2/user/rylansch/FieteLab-Recursive-Nonstationary-CRP/03_mixture_of_vonmises_fisher/results/ 03_mixture_of_vonmises_fisher/results/
rsync -avh --exclude='*.joblib' rylansch@openmind-dtn.mit.edu:/om2/user/rylansch/FieteLab-Recursive-Nonstationary-CRP/05_swav_pretrained/results/ 05_swav_pretrained/results/
rsync -avh --exclude='*.joblib' rylansch@openmind-dtn.mit.edu:/om2/user/rylansch/FieteLab-Recursive-Nonstationary-CRP/07_yilun_nav_2d/results/ 07_yilun_nav_2d/results/
