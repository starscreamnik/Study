using System;

namespace Geometric_figures
{
    class Menu
    {
        public static void MainMenu()
        {
            Console.Clear();
            Console.Write("n - Create new figure\n" +
                "p - Print figure list\n" +
                "h - Help\n" +
                "q - Exit\n");
        }

        public static void SelectFigure()
        {
            Console.Clear();
            Console.Write("e - Ellipse\n" +
                "c - Circle\n" +
                "p - Polygon\n" +
                "q - cancel\n");
        }

        public static void InputCircle()
        {
            Console.Write("x,y,rad: ");
        }

        public static void InputEllipse()
        {
            Console.Write("x,y,a,b: ");
        }

        public static void InputPolygonSize()
        {
            Console.Write("size: ");
        }

        public static void InputPolygonPoint()
        {
            Console.Write("x,y: ");
        }
    }
}