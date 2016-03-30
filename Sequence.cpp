/**
 * thunderstruck/tracker: Sequence.cpp
 */
#include "stdafx.h"
#include "Sequence.h"

#include <exception>
#include <fstream>
#include <iostream>

#include <boost/algorithm/string/trim.hpp>
#include <boost/format.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/regex.hpp>
using boost::format;

namespace thunderstruck	{
//#################### CONSTRUCTORS ####################

Sequence::Sequence(const std::string& sequenceDir)
: m_sequencePath(sequenceDir)
{
  std::cout << "Reading sequence data from " << canonical(m_sequencePath) << '\n';
  //std::cout << "Sequence name: " << name() << '\n';
  m_firstFrameNumber = -1;
  m_frameCount = -1;

  read_frames_file();
  read_ground_truth_file();
}

//#################### PUBLIC MEMBER FUNCTIONS ####################

const cv::Rect_<float>& Sequence::bounding_box(size_t i) const
{
  return m_boundingBoxes[i];
}

cv::Mat Sequence::frame(size_t i) const
{
  std::string filename = str(format("%04d.jpg") % (m_firstFrameNumber + i));
  cv::Mat readFrame = cv::imread((m_sequencePath / filename).string().c_str());
  return readFrame;
}

size_t Sequence::frame_count() const
{
  return m_frameCount;
}

std::string Sequence::name() const
{
  return m_sequencePath.parent_path().leaf().generic_string();
}

//#################### PRIVATE MEMBER FUNCTIONS ####################

void Sequence::read_frames_file()
{
  //boost::filesystem::path framesPath = m_sequencePath / (name() + "_frames.txt");
  boost::filesystem::path framesPath = m_sequencePath / "frames.txt";

  std::ifstream fs(framesPath.c_str());
  if(fs.fail())
  {
    throw std::runtime_error("Could not read sequence frames file");
  }

  std::string line;
  std::getline(fs, line);
  int startFrame = -1;
  int endFrame = -1;
  sscanf(line.c_str(), "%d,%d", &startFrame, &endFrame);
  if (fs.fail() || startFrame == -1 || endFrame == -1)
  {
	  std::cout << "error: could not parse sequence frames file" << std::endl;
	   throw std::runtime_error("Could not read sequence frames file");
  }
  m_firstFrameNumber = (size_t)startFrame;
  m_frameCount = (size_t)(endFrame + 1 - startFrame);
//   boost::trim(line);
//   boost::regex expr("(\\d+),(\\d+)");//
//   boost::smatch what;

//   if(regex_match(line, what, expr))
//   {
//     int startFrame = boost::lexical_cast<int>(what[1]);
//     int endFrame = boost::lexical_cast<int>(what[2]);
//     m_firstFrameNumber = startFrame;
//     m_frameCount = endFrame + 1 - startFrame;
//   }
//   else
//   {
//     throw std::runtime_error("Could not parse sequence frame range");
//   }
}

void Sequence::read_ground_truth_file()
{
  boost::filesystem::path gtPath = m_sequencePath /  + "groundtruth_rect.txt";
  std::ifstream fs(gtPath.c_str());
  if(fs.fail())
  {
    throw std::runtime_error("Could not read sequence ground truth file");
  }

  std::string line;
  std::getline(fs, line);
  float xmin = -1.f;
  float ymin = -1.f;
  float width = -1.f;
  float height = -1.f;

  sscanf(line.c_str(), "%f,%f,%f,%f", &xmin, &ymin, &width, &height);

  if (fs.fail() || xmin < 0.f || ymin < 0.f || width < 0.f || height < 0.f)
  {
	  std::cout << "error: could not parse sequence gt file" << std::endl;
	  throw std::runtime_error("Could not read sequence frames file");
  }
   m_boundingBoxes.push_back(cv::Rect_<float>(xmin, ymin, width, height));

  //while(std::getline(fs, line))
  //{
  //  boost::trim(line);
  //  if(line == "") continue;

  //  std::string real = "(-?\\d+(?:\\.\\d+)?)";
  //  boost::regex expr(real + "," + real + "," + real + "," + real);
  //  boost::smatch what;
  //  if(regex_match(line, what, expr))
  //  {
  //    // The captured values are xMin, yMin, width and height, in that order.
  //    float values[4];
  //    for(size_t i = 0; i < 4; ++i)
  //    {
  //      values[i] = boost::lexical_cast<float>(what[i+1]);
  //    }
  //    m_boundingBoxes.push_back(cv::Rect_<float>(values[0], values[1], values[2], values[3]));
  //  }
  //  else
  //  {
  //    throw std::runtime_error("Could not parse one of the ground truth bounding boxes");
  //  }
  //}
}


}