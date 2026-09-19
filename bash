echo "# Llms.txt" >> README.md
git init
git add README.md
git commit -m "first commit"
git branch -M main
git remote add origin https://github.com/web4hub/web4.0.git
git push -u origin main
pip install llms-txt
