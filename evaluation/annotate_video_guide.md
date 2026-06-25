# Video Annotation Guide

## Valid COCO Classes for Indoor Navigation

Only annotate obstacles that match these YOLO class names:

- `person`
- `chair`
- `couch` (also try `sofa`)
- `dining table`
- `bed`
- `tv`
- `laptop`
- `refrigerator`
- `sink`
- `toilet`
- `bottle`
- `cup`
- `book`

## Annotation Process

1. Open video in VLC or QuickTime
2. Find each moment where an obstacle becomes unavoidable
3. Note the timestamp (seconds with 1 decimal place precision)
4. Create a JSON file with the same name as your video

## Example: `kitchen_walking.json`

```json
{
  "events": [
    {"obstacle": "chair", "critical_time_s": 4.2},
    {"obstacle": "dining table", "critical_time_s": 11.8},
    {"obstacle": "person", "critical_time_s": 18.3}
  ]
}
```

## What is `critical_time_s`?

The **last moment** a spoken warning would still be useful. If the user is walking at normal pace (1.4 m/s), this is typically when the obstacle is ~2-3 meters away.

## Videos with no obstacles

If a video has no clear obstacle moments (e.g., empty hallway), create an empty events list:

```json
{
  "events": []
}
```

This video will only contribute to the false-alert measurement.
