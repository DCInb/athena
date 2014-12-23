//======================================================================================
//! \file blockuid.cpp
//  \brief implementation of functions in the BlockUID class
//  Level is defined as "logical level", where the logical root (single block) is 0,
//  and the physical root (user-specified root level) can be different.
//  Unique ID (+level) gives the absolute location of the block, and can be used to
//  sort the blocks using the Z-ordering.
//======================================================================================


#include "athena.hpp"
#include "blockuid.hpp"
#include <iostream>
#include <sstream>
#include <stdexcept>


//--------------------------------------------------------------------------------------
//! \fn BlockUID& BlockUID::operator=(const BlockUID& bid)
//  \brief override the assignment operator
BlockUID& BlockUID::operator=(const BlockUID& bid)
{
  if (this != &bid){
    level=bid.level;
    for(int i=0;i<IDLENGTH;i++)
      uid[i]=bid.uid[i];
  }
  return *this;
}

//! \fn BlockUID::BlockUID(const BlockUID& bid)
//  \brief copy constructor
BlockUID::BlockUID(const BlockUID& bid)
{
  level=bid.level;
  for(int i=0;i<IDLENGTH;i++)
    uid[i]=bid.uid[i];
}

//--------------------------------------------------------------------------------------
//! \fn bool BlockUID::operator==(const BlockUID& bid)
//  \brief override the comparison (equal) operator
bool BlockUID::operator==(const BlockUID& bid) const
{
  if(level!=bid.level)
  {return false;}
  for(int i=0; i<IDLENGTH; i++)
  {
    if(uid[i]!=bid.uid[i])
    { return false;}
  }
  return true;
}

//--------------------------------------------------------------------------------------
//! \fn bool BlockUID::operator<(const BlockUID& bid)
//  \brief override the comparison (less than) operator
bool BlockUID::operator< (const BlockUID& bid) const
{
  if(level==0)  return true; // the logical root is always the smallest
  for(int i=0; i<IDLENGTH-1; i++)
  {
    if(uid[i]>bid.uid[i])
    { return false;}
    if(uid[i]<bid.uid[i])
    { return true;}
  }
  return uid[IDLENGTH-1]<bid.uid[IDLENGTH-1];
}

//--------------------------------------------------------------------------------------
//! \fn bool BlockUID::&operator> (const BlockUID& bid)
//  \brief override the comparison (larger than) operator
bool BlockUID::operator> (const BlockUID& bid) const
{
  if(level==0)  return true; // the logical root is always the smallest
  for(int i=0; i<IDLENGTH-1; i++)
  {
    if(uid[i]<bid.uid[i])
    { return false;}
    if(uid[i]>bid.uid[i])
    { return true;}
  }
  return uid[IDLENGTH-1]>bid.uid[IDLENGTH-1];
}


//--------------------------------------------------------------------------------------
//! \fn void BlockUID::SetUID(ID_t *suid, int llevel)
//  \brief set the unique ID directly, mainly used for restarting
void BlockUID::SetUID(ID_t *suid, int llevel)
{
  level=llevel;
  for(int i=0;i<IDLENGTH;i++)
    uid[i]=suid[i];
  return;
}


//--------------------------------------------------------------------------------------
//! \fn void BlockUID::CreateUIDfromLocation(long int lx, long int ly, long int lz,
//                                           int llevel)
//  \brief create the unique ID from the location, mainly used for initialization
//         lx, ly, lz are the absolute location in llevel
void BlockUID::CreateUIDfromLocation(long int lx, long int ly, long int lz, int llevel)
{
  long int maxid=1L<<llevel;
  long int bx, by, bz, sh;
  ID_t pack;
  std::stringstream msg;

  if(lx<0 || lx>=maxid || ly<0 || ly>=maxid || lz<0 || lz>=maxid)
  {
    msg << "### FATAL ERROR in CreateUIDfromLocation" << std::endl
        << "The block location is beyond the maximum." << std::endl;
    throw std::runtime_error(msg.str().c_str());
  }
  level=llevel;
  if(llevel==0)  return;

  for(int i=0; i<IDLENGTH; i++)  uid[i]=0;


  for(int l=1;l<=llevel;l++)
  {
    sh=llevel-l;
    bz=((lz >> sh) & 1L) << 2;
    by=((ly >> sh) & 1L) << 1;
    bx=(lx >> sh) & 1L;
    pack=bx | by | bz;
    sh=(usize-(l-1)%usize-1)*3;
    uid[(l-1)/usize] |= (pack << sh);
  }
}

//--------------------------------------------------------------------------------------
//! \fn void BlockUID::CreateUIDbyRefinement(BlockUID coarse, int ox, int oy, int oz)
//  \brief create the unique ID from the ID of the finer level, used for refinement
//         ox, oy, oz indicate the location of the finer level (0=left, 1=right, etc.)
void BlockUID::CreateUIDbyRefinement(BlockUID& coarse, int ox, int oy, int oz)
{
  ID_t pack;
  int sh;
  *this=coarse;
  level++;
  pack = (oz << 2) | (oy << 1) | ox;
  sh=(usize-(level-1)%usize-1)*3;
  uid[(level-1)/usize] |= (pack << sh);
}

//--------------------------------------------------------------------------------------
//! \fn void BlockUID::CreateUIDbyRefinement(BlockUID& fine)
//  \brief create the unique ID from the ID of the finer block, used for derefinement
void BlockUID::CreateUIDbyDerefinement(BlockUID& fine)
{
  ID_t mask=~7;
  int sh;
  *this=fine;
  level--;
  sh=(usize-level%usize-1)*3;
  uid[level/usize] &= (mask << sh);
}


//--------------------------------------------------------------------------------------
//! \fn void BlockUID::GetLocation(long int& lx, long int& ly, long int& lz, int& llevel)
//  \brief get the location from the unique ID
void BlockUID::GetLocation(long int& lx, long int& ly, long int& lz, int& llevel)
{
  ID_t pack;
  long int bx, by, bz;
  int l, sh;
  llevel=level;
  lx=ly=lz=0;
  for(l=1;l<=level;l++)
  {
    sh=(usize-(l-1)%usize-1)*3;
    pack = (uid[(l-1)/usize] >> sh);
    bz=(pack>>2) & 1L;
    by=(pack>>1) & 1L;
    bx=pack & 1L;
    sh=llevel-l;
    lx|=(bx<<sh);
    ly|=(by<<sh);
    lz|=(bz<<sh);
  }
}


//--------------------------------------------------------------------------------------
//! \fn BlockTree::BlockTree()
//  \brief constructor of BlockTree, creates the logical root
BlockTree::BlockTree()
{
  flag=true;
  gid=-1;
  pparent=NULL;
  for(int k=0; k<=1; k++) {
    for(int j=0; j<=1; j++) {
      for(int i=0; i<=1; i++) {
        pleaf[k][j][i]=NULL;
      }
    }
  }
}

//--------------------------------------------------------------------------------------
//! \fn BlockTree::BlockTree(BlockTree *parent, int ox, int oy, int oz)
//  \brief constructor of BlockTree, creates a leaf
BlockTree::BlockTree(BlockTree *parent, int ox, int oy, int oz)
{
  flag=true;
  gid=-1;
  pparent=parent;
  uid.CreateUIDbyRefinement(parent->uid,ox,oy,oz);
  long int lx,ly,lz;
  int ll;
  for(int k=0; k<=1; k++) {
    for(int j=0; j<=1; j++) {
      for(int i=0; i<=1; i++) {
        pleaf[k][j][i]=NULL;
      }
    }
  }
}


//--------------------------------------------------------------------------------------
//! \fn BlockTree::~BlockTree()
//  \brief destructor of BlockTree, destroy all the leaves
BlockTree::~BlockTree()
{
  for(int k=0; k<=1; k++) {
    for(int j=0; j<=1; j++) {
      for(int i=0; i<=1; i++) {
        if(pleaf[k][j][i]!=NULL)
          delete pleaf[k][j][i];
      }
    }
  }
}

//--------------------------------------------------------------------------------------
//! \fn void BlockTree::CreateRootGrid(long int nx, long int ny, long int nz, int nl)
//  \brief create the root grid; the root grid can be incomplete (less than 8 leaves)
void BlockTree::CreateRootGrid(long int nx, long int ny, long int nz, int nl)
{
  long int lx, ly, lz;
  int ll;
  uid.GetLocation(lx, ly, lz, ll);
  if(ll == nl) return;
  for(int k=0; k<=1; k++) {
    if((lz*2+k)*(1L<<(nl-ll-1)) < nz) {
      for(int j=0; j<=1; j++) {
        if((ly*2+j)*(1L<<(nl-ll-1)) < ny) {
          for(int i=0; i<=1; i++) {
            if((lx*2+i)*(1L<<(nl-ll-1)) < nx) {
              flag=false; // if there is a leaf, this is not a leaf
              gid=-1;
              pleaf[k][j][i] = new BlockTree(this, i, j, k);
              pleaf[k][j][i]->CreateRootGrid(nx, ny, nz, nl);
            }
          }
        }
      }
    }
  }
  return;
}

//--------------------------------------------------------------------------------------
//! \fn void BlockTree::Refine(int dim)
//  \brief make finer leaves
void BlockTree::Refine(int dim)
{
  if(dim==1)
  {
    for(int i=0; i<=1; i++) {
      pleaf[0][0][i] = new BlockTree(this, i, 0, 0);
    }
  }
  else if(dim==2) {
    for(int j=0; j<=1; j++) {
      for(int i=0; i<=1; i++) {
        pleaf[0][j][i] = new BlockTree(this, i, j, 0);
      }
    }
  }
  else {
    for(int k=0; k<=1; k++) {
      for(int j=0; j<=1; j++) {
        for(int i=0; i<=1; i++) {
          pleaf[k][j][i] = new BlockTree(this, i, j, k);
        }
      }
    }
  }
  gid=-1;
  flag=false; // this block is not a leaf anymore
  return;
}

//--------------------------------------------------------------------------------------
//! \fn void BlockTree::Derefine(void)
//  \brief delete leaves
void BlockTree::Derefine(void)
{
  for(int k=0; k<=1; k++) {
    for(int j=0; j<=1; j++) {
      for(int i=0; i<=1; i++) {
        if(pleaf[k][j][i]!=NULL)
          delete pleaf[k][j][i];
      }
    }
  }
  flag=true; // this block is now a leaf
  // need to recalculate gid
  return;
}

//--------------------------------------------------------------------------------------
//! \fn void BlockTree::AssignGID(int& id)
//  \brief assign IDs to the leaves and count the total number of the blocks
void BlockTree::AssignGID(int& id)
{
  if(pparent==NULL) // clear id if this is the root of the tree
    id=0;

  if(flag==true)
  {
    gid=id;
    if(myrank==0)
    {
      long lx, ly, lz;
      int ll;
      uid.GetLocation(lx,ly,lz,ll);
    }
    id++;
    return;
  }

  gid==-1;

  for(int k=0; k<=1; k++) {
    for(int j=0; j<=1; j++) {
      for(int i=0; i<=1; i++) {
        if(pleaf[k][j][i]!=NULL)
          pleaf[k][j][i]->AssignGID(id); // depth-first search
      }
    }
  }
  return;
}


//--------------------------------------------------------------------------------------
//! \fn void BlockTree::GetIDList(BlockUID *list, int& count)
//  \brief creates the ID list sorted by Z-ordering
void BlockTree::GetIDList(BlockUID *list, int& count)
{
  if(pparent==NULL) count=0;
  if(flag==true)
  {
    list[count]=uid;
    count++;
  }
  else {
    for(int k=0; k<=1; k++) {
      for(int j=0; j<=1; j++) {
        for(int i=0; i<=1; i++) {
          if(pleaf[k][j][i]!=NULL)
            pleaf[k][j][i]->GetIDList(list, count);
        }
      }
    }
  }
  return;
}


//--------------------------------------------------------------------------------------
//! \fn BlockTree* BlockTree::FindNeighbor(enum direction dir, BlockUID id,
//                            long int rbx, long int rby, long int rbz, int rl)
//  \brief find a neighboring block, called from the root of the tree
//         If it is coarser or same level, return the pointer to that block.
//         If it is a finer block, return the pointer to its parent.
BlockTree* BlockTree::FindNeighbor(enum direction dir, BlockUID id,
                                   long int rbx, long int rby, long int rbz, int rl)
{
  long int lx, ly, lz;
  int ll, level;
  int ox,oy,oz;
  BlockTree *bt = this;
  id.GetLocation(lx,ly,lz,ll);

  if(ll<1) return this; // single grid; return itself

  switch(dir)
  {
    case inner_x1:
      lx--;
      break;
    case outer_x1:
      lx++;
      break;
    case inner_x2:
      ly--;
      break;
    case outer_x2:
      ly++;
      break;
    case inner_x3:
      lz--;
      break;
    case outer_x3:
      lz++;
      break;
    default:
     return NULL;
  }
  // periodic boundaries
  if(lx<0) lx=(rbx<<(ll-rl))-1;
  if(lx>=rbx<<(ll-rl)) lx=0;
  if(ly<0) ly=(rby<<(ll-rl))-1;
  if(ly>=rby<<(ll-rl)) ly=0;
  if(lz<0) lz=(rbz<<(ll-rl))-1;
  if(lz>=rbz<<(ll-rl)) lz=0;

  for(level=0;level<=ll;level++)
  {
    if(bt->flag==true) // leaf
    {
      if(level == ll || level == ll-1)
        return bt;
      else
        return NULL;
    }
    // find a leaf in the next level
    int sh=ll-level-1;
    ox=(lx>>sh) & 1L;
    oy=(ly>>sh) & 1L;
    oz=(lz>>sh) & 1L;
    bt=bt->pleaf[oz][oy][ox];
    if(bt==NULL) return NULL;
  }
  // one level finer: check if they are leaves
  if(bt->pleaf[0][0][0]->flag==true)
    return bt;
  return NULL;
}



//--------------------------------------------------------------------------------------
//! \fn BlockTree* BlockTree::GetIDLeaf(int ox, int oy, int oz)
//  \brief returns the pointer to a leaf
BlockTree* BlockTree::GetLeaf(int ox, int oy, int oz)
{
  return pleaf[oz][oy][ox];
}


//--------------------------------------------------------------------------------------
//! \fn BlockTree* BlockTree::FindNeighbor(enum direction dir, BlockUID id)
//  \brief find a neighboring block, called from the root of the tree
//         If it is coarser or same level, return the pointer to that block.
//         If it is a finer block...
NeighborBlock BlockTree::GetNeighbor(void)
{
  NeighborBlock nei;
  nei.gid=gid;
  nei.level=uid.GetLevel();
  return nei;
}

