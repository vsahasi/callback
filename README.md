# Callback

An agent that makes the phone call to confirm a vendor's bank change.

When a vendor emails new bank details, Callback holds the payment in Ramp, takes the vendor's phone number from the oldest invoice or contract on file from before the change, and calls that number. The vendor says the new account out loud, and Callback never reads it to them. Callback then texts a code to the same line, decides verified, mismatch, or needs review, and attaches the recording and transcript to the bill. A person approves before any money moves.

Built by Veer Sahasi and Eric Catalano for the BBS x Ramp Hackathon.

## Secrets

Never commit API keys. Put them in `.env`, which is gitignored, and list the variable names in `.env.example`.
