# For commit and push the changed files.
git add .

git commit -m "Update:"
git config --global https.proxy http://proxy2.pi.sjtu.edu.cn:3128
git config --global http.proxy http://proxy2.pi.sjtu.edu.cn:3128
git push origin master