all: rayman_min.py

rayman_min.py: shortest_path.py minimax.py Makefile
	cat shortest_path.py > rayman.py
	sed '/from .shortest_path/d' minimax.py >> rayman.py
	sed -i 's/\.\.\./../' rayman.py
	pyminify rayman.py --rename-globals --remove-literal-statements --remove-asserts --remove-debug --output rayman_min.py
