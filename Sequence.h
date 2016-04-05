/**
 * thunderstruck/tracker: Sequence.h
 */

#ifndef H_THUNDERSTRUCK_SEQUENCE
#define H_THUNDERSTRUCK_SEQUENCE

#include <string>
#include <vector>

#include <boost/filesystem.hpp>

#include <opencv2/opencv.hpp>

namespace thunderstruck	{
/**
 * \brief An instance of this class can be used to represent a tracking sequence (a list of frames in which we are to track one or more objects).
 */
class Sequence
{
  //#################### PRIVATE MEMBER FUNCTIONS ####################
private:
  /** The (ground truth) bounding boxes for the frames in the sequence. */
  std::vector<cv::Rect_<float> > m_boundingBoxes;

  /** The number of the first frame in the sequence. */
  size_t m_firstFrameNumber;

  /** The number of frames in the sequence. */
  size_t m_frameCount;

  /** The path to the sequence directory. */
  boost::filesystem::path m_sequencePath;

  //#################### CONSTRUCTORS ####################
public:
  /**
   * \brief Constructs a sequence object to represent the tracking sequence in the specified directory.
   *
   * \param[in] sequenceDir The directory containing the sequence.
   */

	size_t getFrameNum(){return m_frameCount;};
  explicit Sequence(const std::string& sequenceDir);

  //#################### PUBLIC MEMBER FUNCTIONS ####################
public:
  /**
   * \brief Gets the (ground truth) bounding box for the i'th frame in the sequence (indexed from zero).
   *
   * \param[in] i The index of the frame whose bounding box we want to get.
   * \return      The bounding box for the i'th frame in the sequence.
   */
  const cv::Rect_<float>& bounding_box(size_t i) const;

  /**
   * \brief Gets the i'th frame in the sequence (indexed from zero).
   *
   * \param[in] i The index of the frame in the sequence to get.
   * \return      The i'th frame in the sequence.
   */
  cv::Mat frame(size_t i) const;

  /**
   * \brief Gets the number of frames in the sequence.
   *
   * \return  The number of frames in the sequence.
   */
  size_t frame_count() const;

  /**
   * \brief Gets the name of the sequence.
   *
   * \return  The name of the sequence.
   */
  std::string name() const;

  //#################### PRIVATE MEMBER FUNCTIONS ####################
private:
  /**
   * \brief Reads the frames file for the sequence being tracked.
   *
   * The frames file specifies a range that identifies which images form the sequence.
   */
  void read_frames_file();

  /**
   * \brief Reads the ground truth file for the sequence being tracked.
   *
   * The ground truth file contains the bounding boxes for the frames in the sequence.
   */
  void read_ground_truth_file();
};
}

#endif
