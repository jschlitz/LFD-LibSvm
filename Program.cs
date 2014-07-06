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


      //n vs all, 0..9
      //var results = Enumerable.Range(0, 10)
      //  .Select(n => VsAll(n + " vs. all", train, test, KernelHelper.PolynomialKernel(2, 1, 1), 0.01, lf => lf == (double)n ? 1.0 : -1.0))
      //  .ToArray();//force.

      //foreach (var item in results)
      //{
      //  Console.WriteLine(item);
      //}
      var ecvs = Enumerable.Range(-4, 5).ToDictionary(n=>n, _=>new List<double>());

      var blah = new List<int>();
      for (int i = -2; i <= 6; i += 2)
        blah.Add(i);

      var results1v5 = blah
        .Select(d => NvsM("1 vs. 5, c=10^" + d, train,
          test,
          //KernelHelper.PolynomialKernel(2, 1, 1),
          KernelHelper.RadialBasisFunctionKernel(1),
          Math.Pow(10, d),
          lf => lf == 1.0 || lf == 5.0,
          lf => lf == (double)1 ? 1.0 : -1.0))
        .ToArray();
      foreach (var item in results1v5)
      {
        Console.WriteLine(item);
      }
      

      Console.ReadKey(true);


    }

    private static TrainInfo NvsM(string info, string[][] train, string[][] test, Kernel kernel, double c, Func<double, bool> filter, Func<double, double> convertY)
    {
      train = train.Where(sa => filter(double.Parse(sa[0]))).ToArray();
      test = test.Where(sa => filter(double.Parse(sa[0]))).ToArray();

      return VsAll(info, train, test, kernel, c, convertY);
    }

    private static TrainInfo VsAll(string info, string[][] train, string[][] test, Kernel kernel, double c, Func<double, double> convertY)
    {
      var trainProb = MakeProblem(train, convertY);
      var testProb = MakeProblem(test, convertY);
      var machine = new C_SVC(trainProb, kernel, c);

      return new TrainInfo
      {
        Info = info,
        Eout = GetErr(testProb, machine),
        Ein = GetErr(trainProb, machine),
        cvAcc = machine.GetCrossValidationAccuracy(10),
        Machine = machine
      };
    }

    private static double GetErr(svm_problem problem, C_SVC machine)
    {
      var testPredict = problem.x.Select(x => machine.Predict(x)).ToArray();
      var result = testPredict.Zip(problem.y, (pred, real) => pred * real)
        .Where(lf => lf < 0.0)
        .Count();
      return (double) result / (double) problem.l;
    }

    private class TrainInfo
    {
      public string Info;
      public double Ein;
      public double Eout;
      public double cvAcc;
      public override string ToString()
      {
        return string.Format("{0} Ein={1:f3} Eout={2:f3} cvAcc={3:f3}", Info, Ein, Eout, cvAcc);
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
