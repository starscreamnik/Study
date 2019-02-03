using System;
using System.IO;
using System.Text;

namespace Geometric_figures
{
    public class LogInfo
    {
        private static FileStream _logFile;
        private static string _pathForFile = null;
        public static string PathForFile
        {
            get { return File.Exists(_pathForFile)? _pathForFile : null; }
            set
            {
                if (_pathForFile == null)
                {
                    _pathForFile = value;
                    if(File.Exists(_pathForFile)) File.Delete(_pathForFile);
                    _logFile = new FileStream(_pathForFile, FileMode.Create);
                    _logFile.Close();
                }
            }
        }

        public static void WriteToLog(string text)
        {
            if (!File.Exists(_pathForFile))
            {
                Console.WriteLine("Create a log file!");
                return;
            }

            using (var fs = new StreamWriter(_pathForFile))
            {
                fs.WriteLine(text);
            }
        }

        ~LogInfo()
        {
            if (_logFile != null)
            {
                _logFile.Close();
                _logFile = null;
            }
        }
    }
}