using System;
using System.Drawing;
using System.Drawing.Imaging;

namespace AsyncImageFilter
{
    class Program
    {    
        static void Main()
        {
            Console.Write("Введите степень размытия(1 - 25): ");
            int.TryParse(Console.ReadLine(), out var k);
            var filter1 = new MatrixFilter(k);
            
            Console.Write("Введите путь к изображению: ");
            var path = Console.ReadLine();
            var image = new Bitmap(path);

            DateTime start = DateTime.Now;
            var res1 = filter1.RunGaussianBlur(image, false);
            TimeSpan fin = DateTime.Now - start;
            Console.WriteLine("sequence mode: time = {0}", fin.ToString());
            res1.Save("sequence.png", ImageFormat.Png);

            
            start = DateTime.Now;
            var res2 = filter1.RunGaussianBlur(image, true);
            fin = DateTime.Now - start;
            Console.WriteLine("async mode: time = {0}", fin.ToString());
            res2.Save("async.png", ImageFormat.Png);
        }
    }
}