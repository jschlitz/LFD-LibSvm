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

    static svm_node[] DatumToNodes(Datum d)
    {
      return d.x.Select((lf, i) => new svm_node
                  {
                    index = i + 1, //I think that index may need to be 1-based?
                    value = lf
                  }).ToArray();
    }

    static void Main(string[] args)
    {
      //read files
      var pwd = Path.GetDirectoryName(Assembly.GetExecutingAssembly().Location);
      var train = ReadProblem(Path.Combine(pwd, "train.txt"));
      var test = ReadProblem(Path.Combine(pwd, "test.txt"));
      


      var wTarget = (new[] { 0.6, 0.8});
//      var C = 10.0;
//      var gamma = 10.0;
      var w0Target = 0.21; //I don't know how to make this have a bias. What.
      Datum[] classified = { };
      for (int experiment = 0; experiment < EXPERIMENTS; experiment++)
      {
        try
        {
          classified = Generate(TRIALS, w0Target, wTarget);

          var testData = Generate(TRIALS * 10, w0Target, wTarget);
          var problem = new svm_problem
          {
            l = classified.Length,
            x = classified.Select(DatumToNodes).ToArray(), 
            y = classified.Select(d => d.y).ToArray(),
          };


          double goodC = 0.0;
          double goodGamma = 0.0;
          double goodAcc = 0.0;

          for (int i = 0; i < 100; i++)
          {
            classified = Generate(TRIALS, w0Target, wTarget);
            problem = new svm_problem
            {
              l = classified.Length,
              x = classified.Select(DatumToNodes).ToArray(),
              y = classified.Select(d => d.y).ToArray(),
            };
            var machine0 = new C_SVC(problem, KernelHelper.LinearKernel(), double.MaxValue);
          }


          for (double C = 1/64.0; C <= 32768; C *= 2)
          {
            //for (double gamma = .00001; gamma <= 101; gamma *= 10)
            //{
            //  Console.WriteLine("------------------------------");
            //  var machine = new C_SVC(problem, KernelHelper.RadialBasisFunctionKernel(gamma), C);
            //  var acc = machine.GetCrossValidationAccuracy(5);
            //  Console.WriteLine("c={0:f3} gamma={1:f3} for {2:f3}", C, gamma, acc);
            //  if (acc > goodAcc)
            //  {
            //    goodC = C;
            //    goodGamma = gamma;
            //    goodAcc = acc;
            //  }
            //}
            Console.WriteLine("------------------------------");
            var machine = new C_SVC(problem, KernelHelper.LinearKernel(), C);
            //machine.Train();
            var acc = machine.GetCrossValidationAccuracy(5);
            var realAcc = ((double)testData.Where(d => machine.Predict(DatumToNodes(d)) == d.y).Count()) / ((double)testData.Length);
            Console.WriteLine("c={0:f3} acc={1:f3} realAcc={2:f3}", C, acc, realAcc);
            if (realAcc > goodAcc)
            {
              goodC = C;
              goodAcc = realAcc;
            }
          }


          Console.WriteLine("------------------------------");
          Console.WriteLine("c={0:f3} gamma={1:f3} for {2:f3}", goodC, goodGamma, goodAcc);
          //var goodMachine = new C_SVC(problem, KernelHelper.RadialBasisFunctionKernel(goodGamma), goodC);
          var goodMachine = new C_SVC(problem, KernelHelper.LinearKernel(), goodC);
          goodMachine.Train();
          var path = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.CommonApplicationData), "LFD-Svm");
          Directory.CreateDirectory(path);
          path = Path.Combine(path, DateTime.Now.ToString("yyyyMMdd-HHmmss-")+experiment);
          goodMachine.Export(path);
          Console.ReadKey(true);
        }
        catch (Exception ex)
        {
          Console.WriteLine(ex.Message);
        }
      }
    }

    private static svm_problem ReadProblem(string theFile)
    {
      using (var sr = new StreamReader(theFile))
      {
        var fromFile = sr.ReadToEnd()
          .Split('\n')
          .Where(s=>!string.IsNullOrWhiteSpace(s))
          .Select(s => s.Split('\t'))
          .ToArray();
        return new svm_problem
        {
          l = fromFile.Count(),
          y = fromFile.Select(l => double.Parse(l.First())).ToArray(),
          x = fromFile.Select(l => l.Skip(1)
            .Select((a, i) => new svm_node { index = i + 1, value = double.Parse(a) }).ToArray())
            .ToArray()
        };
      }
    }

    private static double[] Norm(double[] wPla)
    {
      var mag = (double)Math.Sqrt(wPla.Sum());
      return wPla.Select(x => x / mag).ToArray();
    }

    private static Datum[] Generate(int count, double w0, double[] w)
    {
      return Enumerable.Range(1, count)
        .Select(_ => GetX())
        .Select(ex => new Datum { x = ex, y = GetY(ex, w0, w) })
        .ToArray();
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

  public struct SVInfo
  {
    public double y;
    public double[] x;
    public double alpha;
    public override string ToString()
    {
      return string.Join(",", x) + "," + y.ToString() + "," + alpha.ToString();
    }
  }

  public struct Datum
  {
    public double y;
    public double[] x;
    public override string ToString()
    {
      return string.Join(",", x) + "," + y.ToString();
    }
  }

}
