import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.List;
import java.util.Random;
import java.util.StringTokenizer;
import java.lang.Math;


public class Perceptron {
	//determines the speed and accuracy at which the NN learns
	static float LEARNING_RATE = (float) 0.05;
	
	//number of training inputs
	static int NUM_INSTANCES = 120;
	
	//number of test inputs
	static int TEST_INSTANCES = 30;
	
	//threshold value
	static double theta = 0;
	
	//inputs in order, Sepal Length, Width, Petal Length, Width
	static float [][] inputs = new float [5][NUM_INSTANCES];
	
	//Error at each output node and hidden node
	static int [] errorO = new int [2];
	
	//Weights and initial weights at each output node, including bias
	static float [][] weights = new float [2][5];
	static float[][] initialWeights = new float [2][5];
	
	//Each instance of inputs is associated with an 2X1 output vector 
	static int [][] outputs = new int [NUM_INSTANCES][2];
	
	//The ANN training guesses
	static int [] guessOutputs = new int [2];
	
	//The 2x1 output vectors of each test input
	static int [][] testOutputs = new int [TEST_INSTANCES][2];
	
	//The 5 inputs of the test data, including bias
	static float[][] testInputs = new float [5][TEST_INSTANCES];
	
	//The guess output vectors used to test the accuracy of the ANN
	static int [][] testGuess = new int [TEST_INSTANCES][2];
	
	//A boolean used to flag which file is being read
	static boolean testflag = false;
	
	//Total test error
	static double sumSquaredError = 0;
	
	//Matrix of the test outputs, used to calculate precision and 
	static int [][] confusionMatrix = new int[3][3];
	static float [] precision = new float [3];
	static float [] recall = new float [3];
	
	public static void main(String args []) throws IOException {
	
		//import training data
		List<String> Train = Files.readAllLines(Paths.get("train.txt"));
		//parse training data into their separate arrays
		stringToArray(Train);
		
		double globalError = 0;
		int i, p, iteration;
		float RMSE = 1;
		Random rand = new Random();
		
		//generate random initial weights
		for (i=0; i<2; i++) {
			for (int f=0; f<5; f++) {
				weights[i][f] = (float) ((rand.nextInt(100)+1)/100.00);
				initialWeights[i][f] = weights[i][f];
			}
		}
		
		iteration = 0;
		//Do while ensures that training runs at least once
		//Each iteration, the perceptron makes a guess, calculates the error, then readjusts the weights to reduce error as much as possible
		do {
			iteration++;
			globalError = 0;
			
			//Perceptron makes a guess using the weights on each input
			for(p = 0; p < NUM_INSTANCES; p++) {
				guessOutputs[0] = calcOutput(weights[0], inputs[0][p], inputs[1][p], inputs[2][p], inputs[3][p]);
				guessOutputs[1] = calcOutput(weights[1], inputs[0][p], inputs[1][p], inputs[2][p], inputs[3][p]);
				
				//calculate output error
				calcOutputErrors(p);
				
				//update the weights
				updateWeights(p);
				
				//calculate the total error
				for (i = 0; i<2; i++)
					globalError += (errorO[i]*errorO[i]);
			}
			//calculate root means squared error. When RMSE = 0, there are no errors and the ANN works perfectly
			RMSE = (float) Math.sqrt(globalError/NUM_INSTANCES);
	
		} //If RMSE = 0, the ANN works perfectly. There needs to be a max number of iterations for ANNs that can't find perfect weights
		while (RMSE != 0 && iteration <= 55);
		
		//This imports the test.txt file and tests the accuracy of the ANN
		testWeights();
		
		//Calculates the precision and recall of the ANN with the test data
		calculatePrecisionRecall(confusionMatrix);
		
		//Writes the output text file 
		writeTextFile("Assignment1_Results");
	}
		
	//This calculates the activation function for each perceptron node. It uses a weighted sum the returns a 1 or -1
		static int calcOutput(float weights[], float SepL, float SepW, float PetL, float PetW) {
		
			float sum = SepL*weights[0] + SepW*weights[1] + PetL*weights[2] + PetW*weights[3] + weights[4];
			if(sum >= theta) {
				return 1;}
			else {
				return -1;}
	}
		//This does all the calculations that test how good the ANN is with the test data
		static void calcTestOutput() {
			int o1,o2;
			double success = 0;
			double suc = 0;
		
			for (int i=0; i<TEST_INSTANCES; i++) {
				//Calculates the guess outputs made by the ANN using the calculated weights
				o1 = calcOutput(weights[0], testInputs[0][i], testInputs[1][i], testInputs[2][i], testInputs[3][i]);
				o2 = calcOutput(weights[1], testInputs[0][i], testInputs[1][i], testInputs[2][i], testInputs[3][i]);
				
				//Stores the outputs in a 2D array
				testGuess[i][0] = o1;
				testGuess[i][1] = o2;
				
				//Calculates success rate and builds the confusion matrix
				if (o1 == testOutputs[i][0] && o2 == testOutputs[i][1]) {
					System.out.println("Test " + i + " success!");
					success += 1;
					
					if (o1 == 1 && o2 == -1) {
					confusionMatrix[0][0] += 1;
						}
					else if (o1 == -1 && o2 == -1) {
						confusionMatrix[1][1] += 1;
					}
					else if (o1 == -1 && o2 == 1) {
						confusionMatrix[2][2] += 1;
					}
				}
				else {
					System.out.println("Test " + i + " fail :(");
					
					if (o1 == 1 && o2 == -1 && testOutputs[i][0] == -1 && testOutputs[i][1] == 1) {
						confusionMatrix[0][2] += 1;
						}
					else if (o1 == 1 && o2 == -1 && testOutputs[i][0] == -1 && testOutputs[i][1] == -1) {
						confusionMatrix[0][1] += 1;
						}
					else if (o1 == -1 && o2 == -1 && testOutputs[i][0] == 1 && testOutputs[i][1] == -1) {
							confusionMatrix[1][0] += 1;
						}
					else if (o1 == -1 && o2 == -1 && testOutputs[i][0] == -1 && testOutputs[i][1] == 1) {
						confusionMatrix[1][2] += 1;
					}
					else if (o1 == -1 && o2 == 1 && testOutputs[i][0] == -1 && testOutputs[i][1] == -1) {
							confusionMatrix[2][1] += 1;
						}
					else if (o1 == -1 && o2 == 1 && testOutputs[i][0] == 1 && testOutputs[i][1] == -1) {
						confusionMatrix[2][0] += 1;
					}
				}
				
				}

			sumSquaredError = 30 - success;
			suc = success;
			success = success/30;
			System.out.println("Success rate = " + success*100 + "%, " + suc);
			}
			
		//Method that calculates the output errors at a given input index
		static void calcOutputErrors(int index) {
			
			for (int i = 0; i<2; i++) {
				errorO[i] = outputs[index][i] - guessOutputs[i];
			}
		}
		
		
		//Updates the weights at each node using the formula w = w = c*e*x
		static void updateWeights(int index) {
			for (int f = 0; f<2; f++) {
				for (int i = 0; i<5; i++) {
					weights[f][i] += errorO[f]*LEARNING_RATE*inputs[i][index];
				}
			}
		}
		
		//Reads the input file list and converts the data into their respective arrays
		static void stringToArray(List<String> Train){

			String[] stringArray = Train.toArray(new String[0]);
			
			for (int i=0; i< NUM_INSTANCES; i++) {
				
				StringTokenizer tokenizer = new StringTokenizer(stringArray[i], ",");
				
				setSepL(i, Float.parseFloat(tokenizer.nextToken()));
				setSepW(i, Float.parseFloat(tokenizer.nextToken()));
				setPetL(i, Float.parseFloat(tokenizer.nextToken()));
				setPetW(i, Float.parseFloat(tokenizer.nextToken()));
				setOutputString(i, tokenizer.nextToken());
			
			}
		}
		
		//constructors that help the stringtoArray method
		static void setSepL(int index, float value) {
			if (testflag == false)
				inputs[0][index] = value;
			else
				testInputs[0][index] = value;
		}
		static void setSepW(int index, float value) {
			if (testflag == false)
				inputs[1][index] = value;
			else
				testInputs[1][index] = value;
		}
		static void setPetL(int index, float value) {
			if (testflag == false)
				inputs[2][index] = value;
			else
				testInputs[2][index] = value;
		}
		static void setPetW(int index, float value) {
			if (testflag == false)
				inputs[3][index] = value;
			else
				testInputs[3][index] = value;
			inputs[4][index] = 1;
		}
		
		//This constructor sets each flower as a 2x1 output vector
		static void setOutputString(int index, String value) {
			//Versicolor has an output vector of [-1,-1]
			if (value.equals("Iris-versicolor")) 
				{
				if(testflag == false) {
					outputs[index][0] = -1;
					outputs[index][1] = -1;}
				else {
					testOutputs[index][0] = -1;
					testOutputs[index][1] = -1;
				}
			}
			//Virginica has an output vector of [-1,1]
			else if (value.equals("Iris-virginica")) 
			{
				if(testflag == false) {
					outputs[index][0] = -1;
					outputs[index][1] = 1;}
				else {
					testOutputs[index][0] = -1;
					testOutputs[index][1] = 1;
				}
			}
			//Setosa has an output vector of [1,-1]
			else if (value.equals("Iris-setosa")) 
			{
				if(testflag == false) {
					outputs[index][0] = 1;
					outputs[index][1] = -1;}
				else {
					testOutputs[index][0] = 1;
					testOutputs[index][1] = -1;
				}
			}
		}
		
		//This reads the test.txt file, parses the strings and sends them to their respective arrays
		static void testWeights() throws IOException {
			testflag = true;
		
			List<String> Test = Files.readAllLines(Paths.get("test.txt"));
			
			String[] stringArray = Test.toArray(new String[0]);
			
			for (int i=0; i< TEST_INSTANCES; i++) {
				
				StringTokenizer tokenizer = new StringTokenizer(stringArray[i], ",");
				
				setSepL(i, Float.parseFloat(tokenizer.nextToken()));
				setSepW(i, Float.parseFloat(tokenizer.nextToken()));
				setPetL(i, Float.parseFloat(tokenizer.nextToken()));
				setPetW(i, Float.parseFloat(tokenizer.nextToken()));
				setOutputString(i, tokenizer.nextToken());
				
			}
			//Calls this method to do all the calculations that test how good the ANN runs
			calcTestOutput();
		
		}
		//This creates the output text file 
		static void writeTextFile(String filename) {
			try {
				BufferedWriter bw = new BufferedWriter(new FileWriter(filename));
				bw.write("Actual Outputs: ");
				bw.newLine();
				
				for (int i = 0; i< testOutputs.length; i++) {
					for (int j = 0; j< testOutputs[i].length; j++) {
						
						bw.write(testOutputs[i][j] + ((j == testOutputs[i].length-1) ? "" : ","));
					}
					bw.newLine();
				}
				bw.write("ANN Outputs: ");
				bw.newLine();
				
				for (int i = 0; i< testGuess.length; i++) {
					for (int j = 0; j< testGuess[i].length; j++) {
						
						bw.write(testGuess[i][j] + ((j == testGuess[i].length-1) ? "" : ","));
					}
					bw.newLine();
				}
				
				bw.write("Initial weights: ");
				bw.newLine();
				
				for (int i = 0; i< initialWeights.length; i++) {
					for (int j = 0; j< initialWeights[i].length; j++) {
						
						bw.write(initialWeights[i][j] + ((j == initialWeights[i].length-1) ? "" : ","));
					}
					bw.newLine();
				}
				
				bw.write("Final weights: ");
				bw.newLine();
				
				for (int i = 0; i< weights.length; i++) {
					for (int j = 0; j< weights[i].length; j++) {
						
						bw.write(weights[i][j] + ((j == weights[i].length-1) ? "" : ","));
					}
					bw.newLine();
				}
				
				bw.write("Total classification Error: " + sumSquaredError);
				bw.newLine();
				
				bw.write("55 Iterations used. The termination criteria is when the root means squared error is equal to 0 or 55 iterations are reached.");
				bw.newLine();
				bw.write("Since this ANN can never be 100% correct, the training stopped at 55 iterations");
				bw.newLine();
				bw.write("Iris-setosa precision = " + precision[0]);
				bw.newLine();
				bw.write("Iris-setosa recall = " + recall[0]);
				bw.newLine();
				bw.write("Iris-versicolor precision = " + precision[1]);
				bw.newLine();
				bw.write("Iris-versicolor recall = " + recall[1]);
				bw.newLine();
				bw.write("Iris-virginica precision = " + precision[2]);
				bw.newLine();
				bw.write("Iris-virginica recall = " + recall[2]);
				bw.newLine();
				
				bw.flush();
			} catch (IOException e) {}
		}
		
		//This method calculates precision and recall
	static void calculatePrecisionRecall(int matrix[][]){
		
		for (int i=0; i<3; i++) {
			precision[i] = (float) matrix[i][i]/(matrix[i][0]+matrix[i][1]+matrix[i][2]);
			recall[i] = (float) matrix[i][i]/(matrix[0][i]+matrix[1][i]+matrix[2][i]);
		}
	}
}
