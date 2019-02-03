using System;

namespace Geometric_figures
{
    public class Menu
    {
        public static void MainMenu()
        {
            Console.Clear(); 
            Console.Write("n - Create new figure\n" +
                          "p - Print figure list\n" +
                          "h - Help\n" +
                          "q - Exit\n");  
            LogInfo.WriteToLog("SCREEN: MainMenu\n");
        }

        public static void SelectFigure()
        {
            Console.Clear(); 
            Console.Write("e - Ellipse\n" +
                          "c - Circle\n" +
                          "p - Polygon\n" +
                          "q - cancel\n");       
            LogInfo.WriteToLog("SCREEN: SelectFigure\n");

        }
    }
}