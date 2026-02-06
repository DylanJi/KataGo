
#include <sstream>
#include <chrono>
#include "../core/global.h"
#include "../core/bsearch.h"
#include "../core/rand.h"
#include "../core/elo.h"
#include "../core/fancymath.h"
#include "../core/config_parser.h"
#include "../core/datetime.h"
#include "../core/fileutils.h"
#include "../core/base64.h"
#include "../core/timer.h"
#include "../core/threadtest.h"
#include "../game/board.h"
#include "../game/rules.h"
#include "../game/boardhistory.h"
#include "../neuralnet/nninputs.h"
#include "../program/gtpconfig.h"
#include "../program/setup.h"
#include "../tests/tests.h"
#include "../tests/tinymodel.h"
#include "../program/playutils.h"
#include "../command/commandline.h"
#include "../main.h"

#include "../external/nlohmann_json/json.hpp"

using namespace std;
using json = nlohmann::json;

// 定义一个获取ownership的请求结构...
struct OWRequest {
    int64_t internalId;
    string id;
    int turnNumber;

    Board board;
    BoardHistory hist;
    Player nextPla;
    int64_t maxVisits;

    static constexpr int STATUS_IN_QUEUE = -1;
    static constexpr int STATUS_POPPED = -2;
    static constexpr int STATUS_TERMINATED = -3;
    std::atomic<int> status;
};


int MainCmds::runownershipobtain(const vector<string>& args){
    cout<< "start a new command, congralations" << endl;

    if(args.size() != 3) {
        cerr << "Must supply exactly two arguments: GTP_CONFIG MODEL_FILE" << endl;
        return 1;
    }

    Board::initHash();
    ScoreValue::initTables();
    Rand seedRand;

    string modelFile;
    string configFile;
    string humanModelFile;
    modelFile = args[1];
    configFile = args[2];
    ConfigParser cfg(configFile);

    cfg.applyAlias("numSearchThreadsPerAnalysisThread", "numSearchThreads");

    auto loadParams = [&humanModelFile](ConfigParser& config, SearchParams& params, Player& perspective, Player defaultPerspective) {
        bool hasHumanModel = humanModelFile != "";
        params = Setup::loadSingleParams(config,Setup::SETUP_FOR_ANALYSIS,hasHumanModel);
        perspective = Setup::parseReportAnalysisWinrates(config,defaultPerspective);
        //Set a default for conservativePass that differs from matches or selfplay
        if(!config.contains("conservativePass"))
            params.conservativePass = true;
    };

    SearchParams defaultParams;
    Player defaultPerspective;
    loadParams(cfg, defaultParams, defaultPerspective, C_EMPTY);

    int numAnalysisThreads;
    numAnalysisThreads = 2;
    int numThreads;
    numThreads = 4;

    const bool logToStdoutDefault = false;
    const bool logToStderrDefault = true;
    Logger logger(&cfg, logToStdoutDefault, logToStderrDefault);

    // 定义一下log的变量
    const bool logAllRequests = cfg.contains("logAllRequests") ? cfg.getBool("logAllRequests") : false;
    const bool logAllResponses = cfg.contains("logAllResponses") ? cfg.getBool("logAllResponses") : false;
    const bool logErrorsAndWarnings = cfg.contains("logErrorsAndWarnings") ? cfg.getBool("logErrorsAndWarnings") : true;
    const bool logSearchInfo = cfg.contains("logSearchInfo") ? cfg.getBool("logSearchInfo") : false;

    const bool assumeMultipleStartingBlackMovesAreHandicap =
        cfg.contains("assumeMultipleStartingBlackMovesAreHandicap") ? cfg.getBool("assumeMultipleStartingBlackMovesAreHandicap") : true;
    const bool preventEncore = cfg.contains("preventCleanupPhase") ? cfg.getBool("preventCleanupPhase") : true;
    
    // 启动一下当前的引擎...
    NNEvaluator* nnEval = NULL;
    {
        Setup::initializeSession(cfg);
        const int expectedConcurrentEvals = numAnalysisThreads * defaultParams.numThreads;
        const bool defaultRequireExactNNLen = false;
        const int defaultMaxBatchSize = -1;
        const bool disableFP16 = false;
        const string expectedSha256 = "";
        nnEval = Setup::initializeNNEvaluator(
        modelFile,modelFile,expectedSha256,cfg,logger,seedRand,expectedConcurrentEvals,
        NNPos::MAX_BOARD_LEN,NNPos::MAX_BOARD_LEN,defaultMaxBatchSize,defaultRequireExactNNLen,disableFP16,
        Setup::SETUP_FOR_ANALYSIS
        );
    }

    ThreadSafeQueue<string*> toWriteQueue;
    auto pushToWrite = [&toWriteQueue](string* s) {
        cout << *s << endl;
        // bool suc = toWriteQueue.forcePush(s);
        // if(!suc)
        //     delete s;
    };

    // Excepted possible keys for queries
    const std::set<std::string> exceptedKeys = {
        "id",
        "turnNumbers",
        "boardXSize",
        "boardYSize",
        "initialStones",
        "moves",
        "rules",
        "maxVisits",
    };

    // 定义一下排队处理的Queue
    ThreadSafePriorityQueue<std::pair<int64_t, int64_t>, OWRequest*> toAnalyzeQueue;
    int64_t internalIdCounter = 0; // Counter for internalId on requests.


    // 输出错误信息
    auto reportError = [&pushToWrite,&logger,&logErrorsAndWarnings](const string& s) {
        json ret;
        ret["error"] = s;
        pushToWrite(new string(ret.dump()));
        if(logErrorsAndWarnings)
            logger.write("Error: " + ret.dump());
    };

    auto reportErrorForId = [&pushToWrite,&logger,&logErrorsAndWarnings](const string& id, const string& field, const string& s) {
        json ret;
        ret["id"] = id;
        ret["field"] = field;
        ret["error"] = s;
        pushToWrite(new string(ret.dump()));
        if(logErrorsAndWarnings)
            logger.write("Error: " + ret.dump());
    };

    auto reportWarningForId = [&pushToWrite,&logger,&logErrorsAndWarnings](const string& id, const string& field, const string& s) {
        json ret;
        ret["id"] = id;
        ret["field"] = field;
        ret["warning"] = s;
        pushToWrite(new string(ret.dump()));
        if(logErrorsAndWarnings)
        logger.write("Warning: " + ret.dump());
    };

    logger.write("Started, ready to begin handling requests");

    // 创建Search 
    Search* bot = new Search(defaultParams, nnEval, &logger, Global::uint64ToString(seedRand.nextUInt64()));

    
    // 获取输入参数
    auto requestLoop = [&](){
        string line;
        json input;
        while (getline(cin, line))
        {
            line = Global::trim(line);
            if (line.length() == 0) {
                continue;
            }
            try {
                input = json::parse(line);
            }
            catch(nlohmann::detail::exception& e){
                reportError(e.what() + string(" - could not parse input line as json request: ") + line);
                continue;
            }

            if (!input.is_object()) {
                reportError("Request line was valid json but was not an object, ignoring: " + input.dump());
                continue;
            }
            
            if (input.find("id") == input.end() || !input["id"].is_string()){
                reportError("Request must have a string \"id\" field");
                continue;
            }

            OWRequest rbase;
            auto parseInteger = [&rbase,&reportErrorForId](const json& dict, const char* field, int64_t& buf, int64_t min, int64_t max, const char* errorMessage) {
                try {
                    if(!dict[field].is_number_integer()) {
                        reportErrorForId(rbase.id, field, errorMessage);
                        return false;
                    }
                    int64_t x = dict[field].get<int64_t>();
                    if(x < min || x > max) {
                        reportErrorForId(rbase.id, field, errorMessage);
                        return false;
                    }
                    buf = x;
                    return true;
                }
                catch(nlohmann::detail::exception& e) {
                    (void)e;
                    reportErrorForId(rbase.id, field, errorMessage);
                    return false;
                }
            };
            
            // parse id
            rbase.id = input["id"].get<string>();
            // parse X and Y
            int boardXSize;
            int boardYSize;
            {
                int64_t xBuf;
                int64_t yBuf;
                static const string boardSizeError = string("Must provide an integer from 2 to ") + Global::intToString(Board::MAX_LEN);
                if (input.find("boardXSize") == input.end()) {
                    reportErrorForId(rbase.id, "boardXSize", boardSizeError.c_str());
                    continue;
                }
                if (input.find("boardYSize") == input.end()) {
                    reportErrorForId(rbase.id, "boardYSize", boardSizeError.c_str());
                    continue;
                }
                if(!parseInteger(input, "boardXSize", xBuf, 2, Board::MAX_LEN, boardSizeError.c_str())) {
                    continue;
                }
                if(!parseInteger(input, "boardYSize", yBuf, 2, Board::MAX_LEN, boardSizeError.c_str())) {
                continue;
                }
                boardXSize = (int)xBuf;
                boardYSize = (int)yBuf;
            }

            auto parseBoardMoves = [boardXSize,boardYSize,&rbase,&reportErrorForId](const json& dict, const char* field, vector<Move>& buf, bool allowPass) {
                buf.clear();
                if(!dict[field].is_array()) {
                    reportErrorForId(rbase.id, field, "Must be an array of pairs of the form: [\"b\" or \"w\", GTP board vertex]");
                    return false;
                }
                for(auto& elt : dict[field]) {
                    if(!elt.is_array() || elt.size() != 2) {
                        reportErrorForId(rbase.id, field, "Must be an array of pairs of the form: [\"b\" or \"w\", GTP board vertex]");
                        return false;
                }

                string s0;
                string s1;
                try {
                    s0 = elt[0].get<string>();
                    s1 = elt[1].get<string>();
                }
                catch(nlohmann::detail::exception& e) {
                    (void)e;
                    reportErrorForId(rbase.id, field, "Must be an array of pairs of the form: [\"b\" or \"w\", GTP board vertex]");
                    return false;
                }

                Player pla;
                if(!PlayerIO::tryParsePlayer(s0,pla)) {
                    reportErrorForId(rbase.id, field, "Could not parse player: " + s0);
                    return false;
                }

                Loc loc;
                if(!Location::tryOfString(s1, boardXSize, boardYSize, loc) ||
                    (!allowPass && loc == Board::PASS_LOC) ||
                    (loc == Board::NULL_LOC)) {
                    reportErrorForId(rbase.id, field, "Could not parse board location: " + s1);
                    return false;
                }
                buf.push_back(Move(loc,pla));
                }
                return true;
            };

            vector<Move> placements;
            if(input.find("initialStones") != input.end()) {
                if(!parseBoardMoves(input, "initialStones", placements, false))
                    continue;
            }
            vector<Move> moveHistory;
            if(input.find("moves") != input.end()) {
                if(!parseBoardMoves(input, "moves", moveHistory, true))
                    continue;
            }
            else {
                reportErrorForId(rbase.id, "moves", "Must specify an array of [player,location] pairs");
                continue;
            }

            // parse analysis turns 
            vector<bool> shouldAnalyze(moveHistory.size()+1,false);
            if(input.find("analyzeTurns") != input.end()) {
                vector<int> analyzeTurns;
                try {
                    analyzeTurns = input["analyzeTurns"].get<vector<int> >();
                }
                catch(nlohmann::detail::exception&) {
                    reportErrorForId(rbase.id, "analyzeTurns", "Must specify an array of integers indicating turns to analyze");
                    continue;
                }

                bool failed = false;
                for(int i = 0; i<analyzeTurns.size(); i++) {
                    int turnNumber = analyzeTurns[i];
                    if(turnNumber < 0 || turnNumber >= shouldAnalyze.size()) {
                        reportErrorForId(rbase.id, "analyzeTurns", "Invalid turn number: " + Global::intToString(turnNumber));
                        failed = true;
                        break;
                    }
                    shouldAnalyze[turnNumber] = true;
                }
                if(failed)
                    continue;
            }
            else {
                shouldAnalyze[shouldAnalyze.size()-1] = true;
            }

            // parse rule
            Rules rules;
            if(input.find("rules") != input.end()) {
                if(input["rules"].is_string()) {
                    string s = input["rules"].get<string>();
                    if(!Rules::tryParseRules(s,rules)) {
                        reportErrorForId(rbase.id, "rules", "Could not parse rules: " + s);
                        continue;
                    }
                }   
                else if(input["rules"].is_object()) {
                    string s = input["rules"].dump();
                    if(!Rules::tryParseRules(s,rules)) {
                        reportErrorForId(rbase.id, "rules", "Could not parse rules: " + s);
                        continue;
                    }
                }
                else {
                    reportErrorForId(rbase.id, "rules", "Must specify rules string, such as \"chinese\" or \"tromp-taylor\", or a JSON object with detailed rules parameters.");
                    continue;
                }
            }
            else {
                reportErrorForId(rbase.id, "rules", "Must specify rules string, such as \"chinese\" or \"tromp-taylor\", or a JSON object with detailed rules parameters.");
                continue;
            }

            // parse max visits 
            if(input.find("maxVisits") != input.end()) {
                bool suc = parseInteger(input, "maxVisits", rbase.maxVisits, 1, (int64_t)1 << 50, "Must be an integer from 1 to 2^50");
                if(!suc)
                    continue;
            }

            Player initialPlayer = C_EMPTY;
            
            // fill init stones...
            Board board(boardXSize, boardYSize);
            for (int i =0; i<placements.size(); i++){
                board.setStone(placements[i].loc, placements[i].pla);
            }

            if (initialPlayer == C_EMPTY) {
                if (moveHistory.size()>0) {
                    initialPlayer = moveHistory[0].pla;
                } else {
                    initialPlayer = BoardHistory::numHandicapStonesOnBoard(board) > 0 ? P_WHITE : P_BLACK;
                }
            }

            // check rule 
            bool rulesWereSupported;
            Rules supportedRules = nnEval->getSupportedRules(rules,rulesWereSupported);
            if(!rulesWereSupported) {
                ostringstream out;
                out << "Rules " << rules << " not supported by neural net, using " << supportedRules << " instead";
                reportWarningForId(rbase.id, "rules", out.str());
                rules = supportedRules;
            }

            // conside next color
            Player nextPla = initialPlayer;
            BoardHistory hist(board,nextPla,rules,0);
            hist.setAssumeMultipleStartingBlackMovesAreHandicap(assumeMultipleStartingBlackMovesAreHandicap);

            bool foundIllegalMove =  false;
            // fill all move his and to the last step...
            for(int i = 0; i<moveHistory.size(); i++) {
                Player movePla = moveHistory[i].pla;
                Loc moveLoc = moveHistory[i].loc;
                bool suc = hist.makeBoardMoveTolerant(board,moveLoc,movePla,preventEncore);
                if(!suc) {
                    reportErrorForId(rbase.id, "moves", "Illegal move " + Global::intToString(i) + ": " + Location::toString(moveLoc,board));
                    foundIllegalMove = true;
                    break;
                }
                nextPla = getOpp(movePla);
            }

            if(foundIllegalMove) {
                continue;
            }

            // 开始计算一下需要的Area...
            Color area[Board::MAX_ARR_SIZE];
            board.calculateArea(area, true, true, true, true);

            // 定义一下输出的信息...
            json res;
            res["id"] = rbase.id;
            res["area"] = json(area);

            // 将res 输出...
            pushToWrite(new string(res.dump()));

            // Create a new request
            
            // for(int turnNumber = moveHistory.size()-1; turnNumber <= moveHistory.size(); turnNumber++) {
            //     if(shouldAnalyze[turnNumber]) {
            //         // OWRequest* newRequest = new OWRequest();
            //         // newRequest->internalId = internalIdCounter++;
            //         // newRequest->id = rbase.id;
            //         // newRequest->turnNumber = turnNumber;
            //         // newRequest->board = board;
            //         // newRequest->hist = hist;
            //         // newRequest->nextPla = nextPla;
            //         // newRequest->status.store(OWRequest::STATUS_IN_QUEUE, std::memory_order_release);
            //         // newRequests.push_back(newRequest);
            //     }
            //     if (turnNumber >= moveHistory.size()) 
            //       break; 
            //     Player movePla = moveHistory[turnNumber].pla;
            //     Loc moveLoc = moveHistory[turnNumber].loc;
            //     if (movePla != nextPla) {
            //         board.clearSimpleKoLoc();
            //         hist.clear(board, movePla, rules, hist.encorePhase);
            //         hist.setAssumeMultipleStartingBlackMovesAreHandicap(assumeMultipleStartingBlackMovesAreHandicap);
            //     }  

            //     bool suc = hist.makeBoardMoveTolerant(board,moveLoc,movePla,preventEncore);
            //     if(!suc) {
            //         reportErrorForId(rbase.id, "moves", "Illegal move " + Global::intToString(turnNumber) + ": " + Location::toString(moveLoc,board));
            //         foundIllegalMove = true;
            //         break;
            //     }
            //     nextPla = getOpp(movePla);
            //     cout << "start to get ownership" << endl;
            //     auto start = std::chrono::high_resolution_clock::now();
            //     vector<double> ownership = PlayUtils::computeOwnership(bot,board,hist,nextPla,rbase.maxVisits);
            //     auto end = std::chrono::high_resolution_clock::now();
            //     std::chrono::duration<double, std::milli> elapsed = end - start;
            //     cout << "Finish get ownership, time cost " << elapsed.count() << endl;
            // }

            if(foundIllegalMove) {
                continue;
            }
        }
        
    };

    Logger::logThreadUncaught("request loop", &logger, requestLoop);

    cout << "here after requestLoop " << endl;

    toAnalyzeQueue.setReadOnly();
    toWriteQueue.setReadOnly();

    return 0;
}