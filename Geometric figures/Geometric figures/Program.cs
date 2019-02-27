using System;
using System.Collections.Generic;
using System.IO;
using Newtonsoft.Json;
using static System.Double;

namespace Geometric_figures
{
    internal class Program
    {

        private static Circle NewCircle()
        {
            Menu.InputCircle();
            var input = Console.ReadLine();
            var split = input.Split(',');
            var x = Parse(split[0]);
            var y = Parse(split[1]);
            var rad = Parse(split[2]);            
            return new Circle(new Point(x,y), rad);
        }

        private static Ellipse NewEllipse()
        {
            Menu.InputEllipse();
            var input = Console.ReadLine();
            var split = input.Split(',');
            var x = Parse(split[0]);
            var y = Parse(split[1]);
            var a = Parse(split[2]);
            var b = Parse(split[3]);
            return new Ellipse(a, b, new Point(x, y));
        }

        private static Polygon NewPolygon()
        {
            Menu.InputPolygonSize();
            var size = Convert.ToInt32(Console.ReadLine());

            var newPolygon = new Point[size];
            for (var i=0; i<newPolygon.Length; i++)
            {
                Menu.InputPolygonPoint();
                var split = Console.ReadLine().Split(',');
                newPolygon[i] = new Point(Parse(split[0]), Parse(split[1]));
            }
            
            return new Polygon(newPolygon);
        }
        private static void CreateFigure(ref List<Shape> figures)
        {
            Menu.SelectFigure();
            var cmd = Console.ReadKey();
            switch (cmd.Key)
            {
                case ConsoleKey.E:
                    Console.Clear();
                    figures.Add(NewEllipse());
                    break;
                case ConsoleKey.C:
                    Console.Clear();
                    figures.Add(NewCircle());
                    break;
                case ConsoleKey.P:
                    Console.Clear();
                    figures.Add(NewPolygon());
                    break;
                case ConsoleKey.Q:
                    break;
                default:
                    CreateFigure(ref figures);
                    break;
            }
        }

        private static void Run()
        {
            var figures = new List<Shape>();
            while (true)
            {
                Menu.MainMenu();

                var cmd = Console.ReadKey(); // Get user input
                
                switch (cmd.Key)
                {
                    case ConsoleKey.N:
                        CreateFigure(ref figures);
                        break;
                    case ConsoleKey.P:
                    {
                        Console.Clear();
                        Console.WriteLine("Type \t\tArea \t\tPerimeter \tCenter");
                        foreach (var figure in figures)
                            Console.WriteLine("{0}\t\t{1}\t\t{2}\t\t{3}",
                                figure.Name,
                                Math.Round(figure.Area(), 3),
                                Math.Round(figure.Perimeter(), 3),
                                figure.Center().GetString(2));

                        Console.ReadKey();
                        break;
                    }
                    case ConsoleKey.H:
                        Console.WriteLine("Help");
                        Console.ReadKey();
                        break;
                    case ConsoleKey.Q:
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