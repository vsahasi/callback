DRAFT for Veer to rewrite in his own words. Not for submission as is.

The voice agent runs the call. A separate detection service decides how much to trust the caller.

The agent is a Python worker built on LiveKit Agents. Deepgram Flux turns the caller's speech into text, GPT-4.1 picks what to say, and Cartesia speaks it. Every tool that reads or changes Ramp goes through one function, the gate.

The detection service joins the LiveKit room as a hidden listener and hears only the caller's audio track. It scores whether the voice is synthetic, whether it matches the enrolled person's voiceprint, and whether the conversation looks like a scam or an AI caller. The agent sends it every finished turn.

The gate is Python code, not a prompt. Before a read or a money move, it asks the service for allow, step_up or deny, and it calls Ramp only on allow. If the service errors, takes over three seconds, or answers for the wrong session, that counts as a deny. On a denied money move the gate holds the vendor and ends the call. Holding never needs verification.

On step_up, the agent runs the factors the service asks for, then asks again. The factors are a random phrase the caller must say back, an SMS code, and a callback. The code and the callback go only to the number on file, never one the caller gives. The service checks the words as well as the voice, so a replayed recording fails.

Voice biometrics alone are broken. In one 2026 study, cloned voices got past a tuned speaker-verification system 43 to 83 percent of the time. So a voiceprint match never authorizes anything alone, and a high deepfake score overrides it.
