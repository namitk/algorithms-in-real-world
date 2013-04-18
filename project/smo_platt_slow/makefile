train: svm-train.o svm-base.o
	g++ $^ -o train

predict: svm-base.o svm-predict.o
	g++ $^ -o predict

%.o: %.cpp
	g++ -c $< -o $@

clean:
	rm -rf *.o train predict