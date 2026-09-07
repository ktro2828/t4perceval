"""ROS 2 message definitions the test bags are written with.

Generated verbatim from the installed ``autoware_perception_msgs`` (autoware_msgs) and ROS 2
Humble common message packages, in the concatenated form an MCAP ``ros2msg`` schema record
holds. The importer decodes a bag from exactly this text, so the tests exercise the same
parser path a real Autoware bag takes.
"""

from __future__ import annotations

__all__ = ("MSGDEFS",)

#: Full datatype name -> concatenated message definition.
MSGDEFS: dict[str, str] = {
    "autoware_perception_msgs/msg/DetectedObjects": r"""std_msgs/Header header
DetectedObject[] objects
================================================================================
MSG: std_msgs/Header
# Standard metadata for higher-level stamped data types.
# This is generally used to communicate timestamped data
# in a particular coordinate frame.

# Two-integer timestamp that is expressed as seconds and nanoseconds.
builtin_interfaces/Time stamp

# Transform frame with which this data is associated.
string frame_id
================================================================================
MSG: autoware_perception_msgs/DetectedObject
float32 existence_probability
ObjectClassification[] classification
DetectedObjectKinematics kinematics
Shape shape
================================================================================
MSG: builtin_interfaces/Time
# This message communicates ROS Time defined here:
# https://design.ros2.org/articles/clock_and_time.html

# The seconds component, valid over all int32 values.
int32 sec

# The nanoseconds component, valid in the range [0, 1e9), to be added to the seconds component.
# e.g.
# The time -1.7 seconds is represented as {sec: -2, nanosec: 3e8}
# The time 1.7 seconds is represented as {sec: 1, nanosec: 7e8}
uint32 nanosec
================================================================================
MSG: autoware_perception_msgs/ObjectClassification
uint8 UNKNOWN = 0
uint8 CAR = 1
uint8 TRUCK = 2
uint8 BUS = 3
uint8 TRAILER = 4
uint8 MOTORCYCLE = 5
uint8 BICYCLE = 6
uint8 PEDESTRIAN = 7
uint8 ANIMAL = 8
uint8 HAZARD = 9 # Defined as an object that can cause danger to autonomous driving
uint8 OVER_DRIVABLE = 10 # Defined as an object that can be safely driven over (e.g., leaf)
uint8 UNDER_DRIVABLE = 11 # Defined as an object that can be safely driven under (e.g., overpass)

# Object classification label (use constants above)
uint8 label
# Classification probability. Range: 0.0 to 1.0
float32 probability
================================================================================
MSG: autoware_perception_msgs/DetectedObjectKinematics
# Only position is available, orientation is empty. Note that the shape can be an oriented
# bounding box but the direction the object is facing is unknown, in which case
# orientation should be empty.
uint8 UNAVAILABLE=0

# The orientation is determined only up to a sign flip. For instance, assume that cars are
# longer than they are wide, and the perception pipeline can accurately estimate the
# dimensions of a car. It should set the orientation to coincide with the major axis, with
# the sign chosen arbitrarily, and use this tag to signify that the orientation could
# point to the front or the back.
uint8 SIGN_UNKNOWN=1

# The full orientation is available. Use e.g. for machine-learning models that can
# differentiate between the front and back of a vehicle.
uint8 AVAILABLE=2

geometry_msgs/PoseWithCovariance pose_with_covariance

bool has_position_covariance
uint8 orientation_availability

geometry_msgs/TwistWithCovariance twist_with_covariance

bool has_twist
bool has_twist_covariance
================================================================================
MSG: autoware_perception_msgs/Shape
# Shape type constants
uint8 BOUNDING_BOX=0  # Use dimensions (x=length, y=width, z=height)
uint8 CYLINDER=1      # Use dimensions (x=diameter, y=diameter, z=height)
uint8 POLYGON=2       # Use footprint as the base polygon with dimensions.z for height

uint8 type
# Footprint polygon (used when type=POLYGON)
geometry_msgs/Polygon footprint
# Dimensions [m] (x, y, z interpretation depends on type)
geometry_msgs/Vector3 dimensions
================================================================================
MSG: geometry_msgs/PoseWithCovariance
# This represents a pose in free space with uncertainty.

Pose pose

# Row-major representation of the 6x6 covariance matrix
# The orientation parameters use a fixed-axis representation.
# In order, the parameters are:
# (x, y, z, rotation about X axis, rotation about Y axis, rotation about Z axis)
float64[36] covariance
================================================================================
MSG: geometry_msgs/TwistWithCovariance
# This expresses velocity in free space with uncertainty.

Twist twist

# Row-major representation of the 6x6 covariance matrix
# The orientation parameters use a fixed-axis representation.
# In order, the parameters are:
# (x, y, z, rotation about X axis, rotation about Y axis, rotation about Z axis)
float64[36] covariance
================================================================================
MSG: geometry_msgs/Polygon
# A specification of a polygon where the first and last points are assumed to be connected

Point32[] points
================================================================================
MSG: geometry_msgs/Vector3
# This represents a vector in free space.

# This is semantically different than a point.
# A vector is always anchored at the origin.
# When a transform is applied to a vector, only the rotational component is applied.

float64 x
float64 y
float64 z
================================================================================
MSG: geometry_msgs/Pose
# A representation of pose in free space, composed of position and orientation.

Point position
Quaternion orientation
================================================================================
MSG: geometry_msgs/Twist
# This expresses velocity in free space broken into its linear and angular parts.

Vector3  linear
Vector3  angular
================================================================================
MSG: geometry_msgs/Point32
# This contains the position of a point in free space(with 32 bits of precision).
# It is recommended to use Point wherever possible instead of Point32.
#
# This recommendation is to promote interoperability.
#
# This message is designed to take up less space when sending
# lots of points at once, as in the case of a PointCloud.

float32 x
float32 y
float32 z
================================================================================
MSG: geometry_msgs/Point
# This contains the position of a point in free space
float64 x
float64 y
float64 z
================================================================================
MSG: geometry_msgs/Quaternion
# This represents an orientation in free space in quaternion form.

float64 x 0
float64 y 0
float64 z 0
float64 w 1
""",
    "autoware_perception_msgs/msg/TrackedObjects": r"""std_msgs/Header header
TrackedObject[] objects
================================================================================
MSG: std_msgs/Header
# Standard metadata for higher-level stamped data types.
# This is generally used to communicate timestamped data
# in a particular coordinate frame.

# Two-integer timestamp that is expressed as seconds and nanoseconds.
builtin_interfaces/Time stamp

# Transform frame with which this data is associated.
string frame_id
================================================================================
MSG: autoware_perception_msgs/TrackedObject
unique_identifier_msgs/UUID object_id
float32 existence_probability
ObjectClassification[] classification
TrackedObjectKinematics kinematics
Shape shape
================================================================================
MSG: builtin_interfaces/Time
# This message communicates ROS Time defined here:
# https://design.ros2.org/articles/clock_and_time.html

# The seconds component, valid over all int32 values.
int32 sec

# The nanoseconds component, valid in the range [0, 1e9), to be added to the seconds component.
# e.g.
# The time -1.7 seconds is represented as {sec: -2, nanosec: 3e8}
# The time 1.7 seconds is represented as {sec: 1, nanosec: 7e8}
uint32 nanosec
================================================================================
MSG: unique_identifier_msgs/UUID
# A universally unique identifier (UUID).
#
#  http://en.wikipedia.org/wiki/Universally_unique_identifier
#  http://tools.ietf.org/html/rfc4122.html

uint8[16] uuid
================================================================================
MSG: autoware_perception_msgs/ObjectClassification
uint8 UNKNOWN = 0
uint8 CAR = 1
uint8 TRUCK = 2
uint8 BUS = 3
uint8 TRAILER = 4
uint8 MOTORCYCLE = 5
uint8 BICYCLE = 6
uint8 PEDESTRIAN = 7
uint8 ANIMAL = 8
uint8 HAZARD = 9 # Defined as an object that can cause danger to autonomous driving
uint8 OVER_DRIVABLE = 10 # Defined as an object that can be safely driven over (e.g., leaf)
uint8 UNDER_DRIVABLE = 11 # Defined as an object that can be safely driven under (e.g., overpass)

# Object classification label (use constants above)
uint8 label
# Classification probability. Range: 0.0 to 1.0
float32 probability
================================================================================
MSG: autoware_perception_msgs/TrackedObjectKinematics
# Only position is available, orientation is empty. Note that the shape can be an oriented
# bounding box but the direction the object is facing is unknown, in which case
# orientation should be empty.
uint8 UNAVAILABLE=0

# The orientation is determined only up to a sign flip. For instance, assume that cars are
# longer than they are wide, and the perception pipeline can accurately estimate the
# dimensions of a car. It should set the orientation to coincide with the major axis, with
# the sign chosen arbitrarily, and use this tag to signify that the orientation could
# point to the front or the back.
uint8 SIGN_UNKNOWN=1

# The full orientation is available. Use e.g. for machine-learning models that can
# differentiate between the front and back of a vehicle.
uint8 AVAILABLE=2

geometry_msgs/PoseWithCovariance pose_with_covariance
geometry_msgs/TwistWithCovariance twist_with_covariance
geometry_msgs/AccelWithCovariance acceleration_with_covariance

uint8 orientation_availability
bool is_stationary
================================================================================
MSG: autoware_perception_msgs/Shape
# Shape type constants
uint8 BOUNDING_BOX=0  # Use dimensions (x=length, y=width, z=height)
uint8 CYLINDER=1      # Use dimensions (x=diameter, y=diameter, z=height)
uint8 POLYGON=2       # Use footprint as the base polygon with dimensions.z for height

uint8 type
# Footprint polygon (used when type=POLYGON)
geometry_msgs/Polygon footprint
# Dimensions [m] (x, y, z interpretation depends on type)
geometry_msgs/Vector3 dimensions
================================================================================
MSG: geometry_msgs/PoseWithCovariance
# This represents a pose in free space with uncertainty.

Pose pose

# Row-major representation of the 6x6 covariance matrix
# The orientation parameters use a fixed-axis representation.
# In order, the parameters are:
# (x, y, z, rotation about X axis, rotation about Y axis, rotation about Z axis)
float64[36] covariance
================================================================================
MSG: geometry_msgs/TwistWithCovariance
# This expresses velocity in free space with uncertainty.

Twist twist

# Row-major representation of the 6x6 covariance matrix
# The orientation parameters use a fixed-axis representation.
# In order, the parameters are:
# (x, y, z, rotation about X axis, rotation about Y axis, rotation about Z axis)
float64[36] covariance
================================================================================
MSG: geometry_msgs/AccelWithCovariance
# This expresses acceleration in free space with uncertainty.

Accel accel

# Row-major representation of the 6x6 covariance matrix
# The orientation parameters use a fixed-axis representation.
# In order, the parameters are:
# (x, y, z, rotation about X axis, rotation about Y axis, rotation about Z axis)
float64[36] covariance
================================================================================
MSG: geometry_msgs/Polygon
# A specification of a polygon where the first and last points are assumed to be connected

Point32[] points
================================================================================
MSG: geometry_msgs/Vector3
# This represents a vector in free space.

# This is semantically different than a point.
# A vector is always anchored at the origin.
# When a transform is applied to a vector, only the rotational component is applied.

float64 x
float64 y
float64 z
================================================================================
MSG: geometry_msgs/Pose
# A representation of pose in free space, composed of position and orientation.

Point position
Quaternion orientation
================================================================================
MSG: geometry_msgs/Twist
# This expresses velocity in free space broken into its linear and angular parts.

Vector3  linear
Vector3  angular
================================================================================
MSG: geometry_msgs/Accel
# This expresses acceleration in free space broken into its linear and angular parts.
Vector3  linear
Vector3  angular
================================================================================
MSG: geometry_msgs/Point32
# This contains the position of a point in free space(with 32 bits of precision).
# It is recommended to use Point wherever possible instead of Point32.
#
# This recommendation is to promote interoperability.
#
# This message is designed to take up less space when sending
# lots of points at once, as in the case of a PointCloud.

float32 x
float32 y
float32 z
================================================================================
MSG: geometry_msgs/Point
# This contains the position of a point in free space
float64 x
float64 y
float64 z
================================================================================
MSG: geometry_msgs/Quaternion
# This represents an orientation in free space in quaternion form.

float64 x 0
float64 y 0
float64 z 0
float64 w 1
""",
    "autoware_perception_msgs/msg/PredictedObjects": r"""std_msgs/Header header
PredictedObject[] objects
================================================================================
MSG: std_msgs/Header
# Standard metadata for higher-level stamped data types.
# This is generally used to communicate timestamped data
# in a particular coordinate frame.

# Two-integer timestamp that is expressed as seconds and nanoseconds.
builtin_interfaces/Time stamp

# Transform frame with which this data is associated.
string frame_id
================================================================================
MSG: autoware_perception_msgs/PredictedObject
unique_identifier_msgs/UUID object_id
float32 existence_probability
ObjectClassification[] classification
PredictedObjectKinematics kinematics
Shape shape
================================================================================
MSG: builtin_interfaces/Time
# This message communicates ROS Time defined here:
# https://design.ros2.org/articles/clock_and_time.html

# The seconds component, valid over all int32 values.
int32 sec

# The nanoseconds component, valid in the range [0, 1e9), to be added to the seconds component.
# e.g.
# The time -1.7 seconds is represented as {sec: -2, nanosec: 3e8}
# The time 1.7 seconds is represented as {sec: 1, nanosec: 7e8}
uint32 nanosec
================================================================================
MSG: unique_identifier_msgs/UUID
# A universally unique identifier (UUID).
#
#  http://en.wikipedia.org/wiki/Universally_unique_identifier
#  http://tools.ietf.org/html/rfc4122.html

uint8[16] uuid
================================================================================
MSG: autoware_perception_msgs/ObjectClassification
uint8 UNKNOWN = 0
uint8 CAR = 1
uint8 TRUCK = 2
uint8 BUS = 3
uint8 TRAILER = 4
uint8 MOTORCYCLE = 5
uint8 BICYCLE = 6
uint8 PEDESTRIAN = 7
uint8 ANIMAL = 8
uint8 HAZARD = 9 # Defined as an object that can cause danger to autonomous driving
uint8 OVER_DRIVABLE = 10 # Defined as an object that can be safely driven over (e.g., leaf)
uint8 UNDER_DRIVABLE = 11 # Defined as an object that can be safely driven under (e.g., overpass)

# Object classification label (use constants above)
uint8 label
# Classification probability. Range: 0.0 to 1.0
float32 probability
================================================================================
MSG: autoware_perception_msgs/PredictedObjectKinematics
geometry_msgs/PoseWithCovariance initial_pose_with_covariance
geometry_msgs/TwistWithCovariance initial_twist_with_covariance
geometry_msgs/AccelWithCovariance initial_acceleration_with_covariance
PredictedPath[] predicted_paths
================================================================================
MSG: autoware_perception_msgs/Shape
# Shape type constants
uint8 BOUNDING_BOX=0  # Use dimensions (x=length, y=width, z=height)
uint8 CYLINDER=1      # Use dimensions (x=diameter, y=diameter, z=height)
uint8 POLYGON=2       # Use footprint as the base polygon with dimensions.z for height

uint8 type
# Footprint polygon (used when type=POLYGON)
geometry_msgs/Polygon footprint
# Dimensions [m] (x, y, z interpretation depends on type)
geometry_msgs/Vector3 dimensions
================================================================================
MSG: geometry_msgs/PoseWithCovariance
# This represents a pose in free space with uncertainty.

Pose pose

# Row-major representation of the 6x6 covariance matrix
# The orientation parameters use a fixed-axis representation.
# In order, the parameters are:
# (x, y, z, rotation about X axis, rotation about Y axis, rotation about Z axis)
float64[36] covariance
================================================================================
MSG: geometry_msgs/TwistWithCovariance
# This expresses velocity in free space with uncertainty.

Twist twist

# Row-major representation of the 6x6 covariance matrix
# The orientation parameters use a fixed-axis representation.
# In order, the parameters are:
# (x, y, z, rotation about X axis, rotation about Y axis, rotation about Z axis)
float64[36] covariance
================================================================================
MSG: geometry_msgs/AccelWithCovariance
# This expresses acceleration in free space with uncertainty.

Accel accel

# Row-major representation of the 6x6 covariance matrix
# The orientation parameters use a fixed-axis representation.
# In order, the parameters are:
# (x, y, z, rotation about X axis, rotation about Y axis, rotation about Z axis)
float64[36] covariance
================================================================================
MSG: autoware_perception_msgs/PredictedPath
geometry_msgs/Pose[] path
builtin_interfaces/Duration time_step
# Range: 0.0 to 1.0
float32 confidence
================================================================================
MSG: geometry_msgs/Polygon
# A specification of a polygon where the first and last points are assumed to be connected

Point32[] points
================================================================================
MSG: geometry_msgs/Vector3
# This represents a vector in free space.

# This is semantically different than a point.
# A vector is always anchored at the origin.
# When a transform is applied to a vector, only the rotational component is applied.

float64 x
float64 y
float64 z
================================================================================
MSG: geometry_msgs/Pose
# A representation of pose in free space, composed of position and orientation.

Point position
Quaternion orientation
================================================================================
MSG: geometry_msgs/Twist
# This expresses velocity in free space broken into its linear and angular parts.

Vector3  linear
Vector3  angular
================================================================================
MSG: geometry_msgs/Accel
# This expresses acceleration in free space broken into its linear and angular parts.
Vector3  linear
Vector3  angular
================================================================================
MSG: builtin_interfaces/Duration
# Duration defines a period between two time points.
# Messages of this datatype are of ROS Time following this design:
# https://design.ros2.org/articles/clock_and_time.html

# The seconds component, valid over all int32 values.
int32 sec

# The nanoseconds component, valid in the range [0, 1e9), to be added to the seconds component.
# e.g.
# The duration -1.7 seconds is represented as {sec: -2, nanosec: 3e8}
# The duration 1.7 seconds is represented as {sec: 1, nanosec: 7e8}
uint32 nanosec
================================================================================
MSG: geometry_msgs/Point32
# This contains the position of a point in free space(with 32 bits of precision).
# It is recommended to use Point wherever possible instead of Point32.
#
# This recommendation is to promote interoperability.
#
# This message is designed to take up less space when sending
# lots of points at once, as in the case of a PointCloud.

float32 x
float32 y
float32 z
================================================================================
MSG: geometry_msgs/Point
# This contains the position of a point in free space
float64 x
float64 y
float64 z
================================================================================
MSG: geometry_msgs/Quaternion
# This represents an orientation in free space in quaternion form.

float64 x 0
float64 y 0
float64 z 0
float64 w 1
""",
    "tf2_msgs/msg/TFMessage": r"""geometry_msgs/TransformStamped[] transforms
================================================================================
MSG: geometry_msgs/TransformStamped
# This expresses a transform from coordinate frame header.frame_id
# to the coordinate frame child_frame_id at the time of header.stamp
#
# This message is mostly used by the
# <a href="https://index.ros.org/p/tf2/">tf2</a> package.
# See its documentation for more information.
#
# The child_frame_id is necessary in addition to the frame_id
# in the Header to communicate the full reference for the transform
# in a self contained message.

# The frame id in the header is used as the reference frame of this transform.
std_msgs/Header header

# The frame id of the child frame to which this transform points.
string child_frame_id

# Translation and rotation in 3-dimensions of child_frame_id from header.frame_id.
Transform transform
================================================================================
MSG: std_msgs/Header
# Standard metadata for higher-level stamped data types.
# This is generally used to communicate timestamped data
# in a particular coordinate frame.

# Two-integer timestamp that is expressed as seconds and nanoseconds.
builtin_interfaces/Time stamp

# Transform frame with which this data is associated.
string frame_id
================================================================================
MSG: geometry_msgs/Transform
# This represents the transform between two coordinate frames in free space.

Vector3 translation
Quaternion rotation
================================================================================
MSG: builtin_interfaces/Time
# This message communicates ROS Time defined here:
# https://design.ros2.org/articles/clock_and_time.html

# The seconds component, valid over all int32 values.
int32 sec

# The nanoseconds component, valid in the range [0, 1e9), to be added to the seconds component.
# e.g.
# The time -1.7 seconds is represented as {sec: -2, nanosec: 3e8}
# The time 1.7 seconds is represented as {sec: 1, nanosec: 7e8}
uint32 nanosec
================================================================================
MSG: geometry_msgs/Vector3
# This represents a vector in free space.

# This is semantically different than a point.
# A vector is always anchored at the origin.
# When a transform is applied to a vector, only the rotational component is applied.

float64 x
float64 y
float64 z
================================================================================
MSG: geometry_msgs/Quaternion
# This represents an orientation in free space in quaternion form.

float64 x 0
float64 y 0
float64 z 0
float64 w 1
""",
}
