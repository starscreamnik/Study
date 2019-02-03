using System;
using System.Collections.Generic;
using System.IO;

namespace Geometric_figures
{
    internal class Program
    {
        private static readonly Random Rand = new Random();
        private static void CreateFigure(ref List<Shape> figures)
        {
            Menu.SelectFigure();
            var cmd = Console.Read();
            switch (cmd)
            {
                case 'e':
                    figures.Add(new Ellipse(1, 2, 3, 4));
                    LogInfo.WriteToLog("CREATE: Ellipse\n");
                    break;
                case 'c':
                    figures.Add(new Circle(10, 2, 3));
                    LogInfo.WriteToLog("CREATE: Circle\n");
                    break;
                case 'p':
                    var p = new Point[Rand.Next(3, 10)];
                    for (var i = 0; i < p.Length; i++)
                    {
                        p[i].X = Rand.Next(100);
                        p[i].Y = Rand.Next(100);
                    }

                    figures.Add(new Polygon(p));
                    LogInfo.WriteToLog("CREATE: Polygon\n");

                    break;
                case 'q':
                    LogInfo.WriteToLog("CREATE: break\n");
                    break;
                default:
                    CreateFigure(ref figures);
                    break;
            }
        }

        private static void Run()
        {
            var figures = new List<Shape>();
            LogInfo.PathForFile = "LOG1.txt";
            LogInfo.WriteToLog("RUN THE PROGRAM\n");
            while (true)
            {
                Menu.MainMenu();
                var cmd = Console.Read();

                switch (cmd)
                {
                    case 'n':
                        CreateFigure(ref figures);
                        break;

                    case 'p':
                        Console.WriteLine("Area \t\tPerimeter \tCenter");
                        foreach (var figure in figures)
                            Console.WriteLine("{0}\t\t{1}\t\t{2}", 
                                figure.Area(), figure.Perimeter(), figure.Center().GetString());
                        LogInfo.WriteToLog("SCREEN: Shape's List");

                        Console.ReadKey();
                        break;

                    case 'h':
                        Console.WriteLine("Help");
                        LogInfo.WriteToLog("SCREEN: Help");
                        Console.ReadKey();
                        break;
                    case 'q':
                        LogInfo.WriteToLog("END OF THE PROGRAM\n");
                        return;
                }
            }
        }


        public static void Main(string[] args)
        {
            Console.WriteLine("Geometry figures");
            Run();
        }
    }
}