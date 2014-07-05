using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using libsvm;
using System.IO;
using System.Reflection;

namespace LFD_LibSvm
{
  class Program
  {
    static Random R = new Random();//(526);
    const int TRIALS = 100;
    const int EXPERIMENTS = 1;
    const int THRESH = 100000000;
    static readonly double sr10 = Math.Sqrt(10);


    static void Main(string[] args)
    {
      //read files
      var pwd = Path.GetDirectoryName(Assembly.GetExecutingAssembly().Location);
      var train = ReadFile(Path.Combine(pwd, "train.txt"));
      var test = ReadFile(Path.Combine(pwd, "test.txt"));


      var results = Enumerable.Range(0, 10)
        .Select(n => VsAll(n + " vs. all", train, test, KernelHelper.PolynomialKernel(2, 1, 1), 0.01, lf => lf == (double)n ? 1.0 : -1.0))
        .ToArray();//force.

      foreach (var item in results)
      {
        Console.WriteLine(item);
      }

      Console.ReadKey(true);


      //var wTarget = (new[] { 0.6, 0.8});
      //var w0Target = 0.21; //I don't know how to make this have a bias. What.
      //Datum[] classified = { };
      //for (int experiment = 0; experiment < EXPERIMENTS; experiment++)
      //{
      //  try
      //  {
      //    classified = Generate(TRIALS, w0Target, wTarget);

      //    var testData = Generate(TRIALS * 10, w0Target, wTarget);
      //    var problem = new svm_problem
      //    {
      //      l = classified.Length,
      //      x = classified.Select(DatumToNodes).ToArray(), 
      //      y = classified.Select(d => d.y).ToArray(),
      //    };


      //    double goodC = 0.0;
      //    double goodGamma = 0.0;
      //    double goodAcc = 0.0;

      //    for (int i = 0; i < 100; i++)
      //    {
      //      classified = Generate(TRIALS, w0Target, wTarget);
      //      problem = new svm_problem
      //      {
      //        l = classified.Length,
      //        x = classified.Select(DatumToNodes).ToArray(),
      //        y = classified.Select(d => d.y).ToArray(),
      //      };
      //      var machine0 = new C_SVC(problem, KernelHelper.LinearKernel(), double.MaxValue);
      //    }


      //    for (double C = 1/64.0; C <= 32768; C *= 2)
      //    {
      //      Console.WriteLine("------------------------------");
      //      var machine = new C_SVC(problem, KernelHelper.LinearKernel(), C);
      //      //machine.Train();
      //      var acc = machine.GetCrossValidationAccuracy(5);
      //      var realAcc = ((double)testData.Where(d => machine.Predict(DatumToNodes(d)) == d.y).Count()) / ((double)testData.Length);
      //      Console.WriteLine("c={0:f3} acc={1:f3} realAcc={2:f3}", C, acc, realAcc);
      //      if (realAcc > goodAcc)
      //      {
      //        goodC = C;
      //        goodAcc = realAcc;
      //      }
      //    }


      //    Console.WriteLine("------------------------------");
      //    Console.WriteLine("c={0:f3} gamma={1:f3} for {2:f3}", goodC, goodGamma, goodAcc);
      //    //var goodMachine = new C_SVC(problem, KernelHelper.RadialBasisFunctionKernel(goodGamma), goodC);
      //    var goodMachine = new C_SVC(problem, KernelHelper.LinearKernel(), goodC);
      //    goodMachine.Train();
      //    var path = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.CommonApplicationData), "LFD-Svm");
      //    Directory.CreateDirectory(path);
      //    path = Path.Combine(path, DateTime.Now.ToString("yyyyMMdd-HHmmss-")+experiment);
      //    goodMachine.Export(path);
      //    Console.ReadKey(true);
      //  }
      //  catch (Exception ex)
      //  {
      //    Console.WriteLine(ex.Message);
      //  }
      //}
    }

    private static TrainInfo VsAll(string info, string[][] train, string[][] test, Kernel kernel, double c, Func<double, double> convertY)
    {
      var trainProb = MakeProblem (train,convertY);
      var machine = new C_SVC(trainProb, kernel, c);
      var predictions = trainProb.x.Select(x => machine.Predict(x)).ToArray();
      var wrong1 =predictions.Zip(trainProb.y, (pred, real) => pred * real)
        .Where(lf => lf < 0.0)
        .Count();
      return new TrainInfo { Info = info, Ein = (double)wrong1 / (double)trainProb.l, Machine = machine };
    }

    private class TrainInfo
    {
      public string Info;
      public double Ein;
      public double Eout;
      public override string ToString()
      {
        return string.Format("{0} Ein={1:f3} Eout={2:f3}", Info, Ein, Eout);
      }
      public C_SVC Machine;
      public void ExportMachine(string name)
      {
        var path = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.CommonApplicationData), "LFD-Svm");
        Directory.CreateDirectory(path);
        path = Path.Combine(path, name + ".xml");
        Machine.Export(path);
      }
    }

    private static string[][] ReadFile(string theFile)
    {
      using (var sr = new StreamReader(theFile))
      {
        return sr.ReadToEnd()
          .Split('\n')
          .Where(s => !string.IsNullOrWhiteSpace(s))
          .Select(s => s.Split('\t'))
          .ToArray();
      }
    }

    private static svm_problem MakeProblem(string[][] raw, Func<double,double> convertY)
    {
      return new svm_problem
      {
        l = raw.Count(),
        y = raw.Select(l => convertY(double.Parse(l.First()))).ToArray(),
        x = raw.Select(l => l.Skip(1)
          .Select((a, i) => new svm_node { index = i + 1, value = double.Parse(a) }).ToArray())
          .ToArray()
      };
    }


    private static double[] Norm(double[] wPla)
    {
      var mag = (double)Math.Sqrt(wPla.Sum());
      return wPla.Select(x => x / mag).ToArray();
    }


    private static double GetY(double[] x, double w0, double[] w)
    {
      return (Dot(x, w) + w0 < 0) ? -1.0 : 1.0;
    }

    private static double Dot(double[] v1, double[] v2)
    {
      return v1.Zip(v2, (v1n, v2n) => v1n * v2n).Aggregate((acc, r) => acc + r);
    }

    private static double[] GetX()
    {
      return new[] { GetR(), GetR() };
    }

    private static double GetR()
    {
      return R.NextDouble() * 2.0 - 1.0;
    }
  }


}
