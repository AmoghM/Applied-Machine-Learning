import java.util.*;
class LogicalAndPerceptron
{
	public static void main(String args[])
	{
		int train_data[][]={{1,0,0,0},{1,0,1,0},{1,1,1,1},{1,1,0,0}};
		double learning_rate=0.1;
		int n=500; //number of iterations
		int i=1,j; 
		double w[]=new double[3];
		w[0]=Math.random();
		w[1]=Math.random(); //random weight 1
		w[2]=Math.random(); // random weight 2
		
		while(i<=n)
		{
			j=0+(int)(Math.random()*3);
			double dot_product=w[0]*train_data[j][0]+w[1]*train_data[j][1]+w[2]*train_data[j][2]; //dot product sigma(w(i),x(i)) for i=1,2
			int observed_value=result(dot_product);
			int expected_value=train_data[j][3];
			int error=expected_value-observed_value;
			w[0]=w[0]+learning_rate*error*train_data[j][0];
			w[1]=w[1]+learning_rate*error*train_data[j][1];
			w[2]=w[2]+learning_rate*error*train_data[j][2];	
		i++;
		}
		
		for(i=0;i<4;i++)
		{
			double dot_product=w[0]*train_data[i][0]+w[1]*train_data[i][1]+w[2]*train_data[i][2];
			int observed_value=result(dot_product);
			int expected_value=train_data[i][3];
			System.out.println("  observed  expected");
			System.out.println("\t"+observed_value+"\t"+expected_value);
			
			
		}
		System.out.println(w[0]+" "+w[1]+" "+" " +w[2]);
	}
	
	public static int result(double prod)
	{
		if(prod<=0)
			return 0;
		else
			return 1;
	}
}
