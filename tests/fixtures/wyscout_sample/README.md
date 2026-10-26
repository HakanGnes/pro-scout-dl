# Wyscout-format sample (fictional)

A tiny, invented dataset (3 teams, 9 players, 2 matches, 39 events) in exactly the
format of the Wyscout public dataset, including its quirks: double-escaped names,
`"null"` strings, `playerId: 0`, empty `subEventId`, single-position events, a player id
missing from `players.json`, goalkeeper events tagged as goals, a penalty and an own goal.

Names, teams and venues are made up. It is used only by the tests.
