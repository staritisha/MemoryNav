# MemoryNav User Study Notes

**Date:** ___________  
**Participant ID:** P___  
**Age range:** ___  
**Relevant condition (if any):** ___________________  
**Evaluator:** ___________

---

## Task Protocol (15 minutes per participant)

### Setup (2 min)
1. Open MemoryNav dashboard on laptop
2. Start webcam feed
3. Brief participant: *"This system uses your camera to detect obstacles and speak warnings. Walk normally, ignore the screen, just listen to the audio."*

### Task 1: Empty hallway walk (2 min)
Participant walks a known route with no obstacles.  
**What we measure:** False alerts (should be 0 or near 0)

| Metric | Observed |
|---|---|
| Alerts fired | |
| User commented on noise | Y / N |

### Task 2: Obstacle course walk (5 min)
Place 3 obstacles: a chair, a bag on the floor, a box.  
Participant walks toward them from 3m.

| Obstacle | Warned before contact? | Warning lead time (seconds) | User reaction |
|---|---|---|---|
| Chair | Y / N | | |
| Bag | Y / N | | |
| Box | Y / N | | |

### Task 3: Memory test (3 min)
Ask participant to add a home note ("there is a step at the kitchen entrance").  
Then walk toward a simulated obstacle while the context note is active.

**Did memory context fire (shown in dashboard)?** Y / N  
**Did risk score change noticeably?** Y / N

### Task 4: Voice query (2 min)
Ask participant to say: "What is in front of me?"  
**Was the answer relevant?** Y / N  
**Latency felt acceptable (<2s)?** Y / N

---

## Post-Task Questions (3 min)

On a scale of 1–5 (1=strongly disagree, 5=strongly agree):

| Question | Score |
|---|---|
| The voice alerts were useful | |
| The alerts were not too frequent | |
| I would trust this system outdoors | |
| I would use this if I had mobility difficulties | |
| The system felt like it understood my home | |

**Open feedback:**  
_______________________________________________  
_______________________________________________  

---

## Notes for README

After running 5 participants, compute:
- Average warning rate (Task 2)
- Average false alerts (Task 1)  
- Average satisfaction score (Post-task mean)

Add to README section 7 (Results) as:
> Informal user study: N=5 participants, X/3 obstacles warned, mean satisfaction Y/5.
