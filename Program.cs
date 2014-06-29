using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace LFD_LibSvm
{
  class Program
  {
    static Random R = new Random();
    const int TRIALS = 100;
    const int EXPERIMENTS = 100;
    const int THRESH = 100000000;

    static void Main(string[] args)
    {
      var wTarget = (new[] { 0.6, 0.8});
      var C = 10;
      var w0Target = 0.1;
      Datum[] classified = { };
      for (int experiment = 0; experiment < EXPERIMENTS; experiment++)
      {
        try
        {
          classified = Generate(TRIALS, w0Target, wTarget);
          var testData = Generate(TRIALS * 10, w0Target, wTarget);
        }
        catch (Exception ex)
        {
          Console.WriteLine(ex.Message);
        }
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
